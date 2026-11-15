import numpy as np
import pytest

from kvwear.codecs import make_codec
from kvwear.policy import ContinuationModel, Policy, RequestContext, ReuseAware, make_policy
from kvwear.store import TieredKVStore, prepare_offload_dir

BLOCK = np.ones((2, 2, 1, 4, 8), np.float16)       # 256 bytes


def store(tmp_path, policy, ram_blocks=2, ssd_blocks=100, future=None, async_writes=False):
    pol = make_policy(policy, future=future or {}) if isinstance(policy, str) else policy
    return TieredKVStore(pol, tmp_path / "off", ram_bytes=ram_blocks * BLOCK.nbytes,
                         ssd_bytes=ssd_blocks * BLOCK.nbytes, async_writes=async_writes)


def ctx(i, conv=0, turn=1):
    return RequestContext(i, conv, turn, False)


def put_all(s, hashes, i=0, conv=0):
    """Same order as the runner: begin -> lookup -> put the blocks that weren't found."""
    s.begin_request(ctx(i, conv), hashes)
    found = len(s.lookup_prefix(hashes))
    for k, h in enumerate(hashes):
        if k < found:
            continue
        s.put(h, BLOCK * (k + 1))


def test_write_through_writes_every_unique_block_once(tmp_path):
    s = store(tmp_path, "write_through")
    put_all(s, ["a", "b", "c"])
    put_all(s, ["a", "b", "c", "d"], i=1)
    s.close()
    assert s.ssd.stats.writes == 4
    assert len(list((tmp_path / "off").rglob("*.kvb"))) == 4


def test_spill_writes_only_on_eviction(tmp_path):
    s = store(tmp_path, "spill", ram_blocks=2)
    put_all(s, ["a", "b"])
    assert s.ssd.stats.writes == 0
    put_all(s, ["c"], i=1)                          # evicts "a"
    s.close()
    assert s.ssd.stats.writes == 1 and "a" in s.ssd


def test_no_ssd_never_writes(tmp_path):
    s = store(tmp_path, "no_ssd", ram_blocks=1)
    put_all(s, list("abcdef"))
    s.close()
    assert s.ssd.stats.writes == 0 and s.stats.ssd_dropped == 5


def test_lookup_returns_longest_prefix_and_promotes(tmp_path):
    s = store(tmp_path, "write_through", ram_blocks=1)
    put_all(s, ["a", "b", "c"])
    got = s.lookup_prefix(["a", "b", "x", "c"])
    s.close()
    assert len(got) == 2 and s.stats.ssd_hits >= 1
    np.testing.assert_array_equal(got[0], BLOCK * 1)
    np.testing.assert_array_equal(got[1], BLOCK * 2)


def test_ssd_capacity_evicts_files(tmp_path):
    s = store(tmp_path, "write_through", ssd_blocks=3)
    put_all(s, list("abcdef"))
    s.close()
    assert len(list((tmp_path / "off").rglob("*.kvb"))) <= 3 and s.ssd.stats.evictions >= 3


def test_async_writer(tmp_path):
    s = store(tmp_path, "write_through", async_writes=True)
    put_all(s, list("abcdefgh"))
    assert len(s.lookup_prefix(list("abcdefgh"))) == 8   # served from RAM or the pending queue
    s.close()
    assert s.ssd.stats.writes == 8 and not s.ssd.pending


def test_quantized_policy_writes_fewer_bytes(tmp_path):
    big = np.random.default_rng(0).standard_normal((2, 2, 2, 16, 64)).astype(np.float16)
    res = {}
    for p in ("write_through", "write_through_int4"):
        s = TieredKVStore(make_policy(p), tmp_path / p, ram_bytes=10**9, ssd_bytes=10**9, async_writes=False)
        s.begin_request(ctx(0), ["a"])
        s.put("a", big)
        res[p] = s.ssd.stats.bytes_written
    assert res["write_through_int4"] < 0.3 * res["write_through"]


def test_reuse_aware_drops_dead_and_keeps_shared(tmp_path):
    adm = ReuseAware(threshold=0.5, model=ContinuationModel(prior=0.9))
    s = store(tmp_path, Policy("reuse", "reuse@0.5/fp16", adm, make_codec("fp16")), ram_blocks=1)
    s.begin_request(ctx(0, conv=0), ["sys", "a"])
    s.put("sys", BLOCK)
    s.put("a", BLOCK)                                    # evicts "sys": conv 0 live, p=0.9 -> written
    s.begin_request(ctx(1, conv=1), ["sys", "b"])        # "sys" now shared by 2 conversations
    s.begin_request(ctx(2, conv=0, turn=2), ["sys", "z"])  # conv 0's context no longer holds "a"
    s.put("z", BLOCK)                                    # evicts "a" -> dead -> dropped
    s.close()
    assert "sys" in s.ssd and "a" not in s.ssd
    assert adm.score("sys") == 1.0 and adm.score("a") == 0.0


def test_ghost_list_proves_blocks():
    adm = ReuseAware(threshold=0.99, model=ContinuationModel(prior=0.1))
    adm.observe_request(ctx(0), ["a"])
    assert not adm.should_write("a")
    adm.on_drop("a")
    adm.on_lookup("a", hit=False)                        # it came back: the filter was wrong
    assert adm.should_write("a")


def test_oracle_writes_only_reused_blocks(tmp_path):
    s = store(tmp_path, "oracle", ram_blocks=0, future={"a": [0, 5], "b": [0]})
    put_all(s, ["a", "b"])
    s.close()
    assert "a" in s.ssd and "b" not in s.ssd


def test_oracle_keeps_blocks_the_current_lookup_still_needs(tmp_path):
    s = store(tmp_path, "oracle", ram_blocks=1, future={"a": [1], "b": [1]})
    s.adm.observe_request(ctx(1), ["a", "b"])            # request 1 is reading its prefix
    assert s.adm.should_write("b")
    s.adm.on_lookup_done()
    assert not s.adm.should_write("b")


def test_refuses_to_wipe_foreign_dir(tmp_path):
    d = tmp_path / "mine"
    d.mkdir()
    (d / "thesis.docx").write_text("important")
    with pytest.raises(RuntimeError):
        prepare_offload_dir(d)
    assert (d / "thesis.docx").exists()
    ok = prepare_offload_dir(tmp_path / "fresh")
    (ok / "junk.kvb").write_text("x")
    prepare_offload_dir(ok)                              # its own folder: wiped
    assert not (ok / "junk.kvb").exists()


def test_policy_specs():
    p = make_policy("reuse@0.3/int8+zlib")
    assert p.admission.threshold == 0.3 and p.codec.name == "int8+zlib"
    with pytest.raises(ValueError):
        make_policy("bogus/fp16")
    with pytest.raises(ValueError):
        make_policy("oracle")


def test_continuation_model_bayes():
    m = ContinuationModel({(False, 1): (10, 8), (False, 2): (8, 1)}, gaps=[1, 2, 2, 3, 4])
    assert m.p_cont(1, False) > 0.7 > m.p_cont(2, False)
    assert m.p_return(1, False, idle=0) == pytest.approx(m.p_cont(1, False))
    assert m.p_return(1, False, idle=10) < 0.1          # idle far longer than any observed gap
    assert m.p_cont(9, False) == 0.05                   # longer than all training conversations
