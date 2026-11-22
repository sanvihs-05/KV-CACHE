"""CUDA kernels: bit-exact against the CPU codec, batched assembly, store integration.

Skipped automatically on machines without a GPU or without nvcc."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from kvwear import cuda as kvcuda  # noqa: E402

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and kvcuda.available()),
                                reason="needs a CUDA GPU and nvcc")

from kvwear.codecs import Codec, EncodedBlock  # noqa: E402


def _block(shape, seed=0, scale=2.0):
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal(shape) * scale).astype(np.float16)
    a.reshape(-1, shape[-1])[0] = 0                     # an all-zero group (scale 0 -> 1)
    a.reshape(-1, shape[-1])[1] = np.float16(0.5)       # a constant group (max == min)
    a.reshape(-1)[7] = np.float16(65504)                # extreme value
    return a


SHAPES = [(2, 2, 4, 64, 64), (1, 2, 2, 64, 128), (3, 2, 1, 8, 32), (1, 2, 1, 3, 16), (1, 1, 1, 5, 256)]


@pytest.mark.parametrize("method", ["int8", "int4"])
@pytest.mark.parametrize("variant", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("shape", SHAPES)
def test_encode_is_byte_identical_to_cpu(method, variant, shape):
    arr = _block(shape, seed=variant)
    cpu = Codec(method).encode(arr)
    gpu = Codec(method, device="cuda")
    gpu._gpu.variant = variant
    assert gpu.encode(arr) == cpu


@pytest.mark.parametrize("method", ["int8", "int4"])
@pytest.mark.parametrize("shape", SHAPES)
def test_decode_is_identical_to_cpu(method, shape):
    arr = _block(shape, seed=3)
    data = Codec(method).encode(arr)
    blk = EncodedBlock(bytearray(data))
    payload = torch.frombuffer(bytearray(blk.payload), dtype=torch.uint8).cuda()
    out = kvcuda.GpuCodec.dequantize_device(payload, arr.size, blk.group, method)
    np.testing.assert_array_equal(out.cpu().numpy().reshape(shape), Codec.decode(data))


def test_compressed_codec_roundtrip_on_gpu():
    arr = _block((2, 2, 2, 64, 64), seed=5)
    a = Codec("int8", compress="zlib").encode(arr)
    b = Codec("int8", compress="zlib", device="cuda").encode(arr)
    assert a == b


@pytest.mark.parametrize("ttot_blocks", [1, 5])
def test_block_loader_matches_cpu_concatenate(ttot_blocks):
    shape = (3, 2, 4, 16, 64)
    blocks, ref = [], []
    for i in range(ttot_blocks):
        arr = _block(shape, seed=10 + i)
        kind = ["fp16_ram", "int8", "int4", "fp16_ssd"][i % 4]
        if kind == "fp16_ram":
            blocks.append(arr)
            ref.append(arr)
        else:
            m = "fp16" if kind == "fp16_ssd" else kind
            data = bytearray(Codec(m).encode(arr))
            blocks.append(EncodedBlock(data))
            ref.append(Codec.decode(data))
    out = kvcuda.BlockLoader().assemble(blocks)
    torch.cuda.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy(), np.concatenate(ref, axis=3))


def test_block_loader_with_zstd_blocks():
    pytest.importorskip("zstandard")
    shape = (2, 2, 2, 16, 64)
    arrs = [_block(shape, seed=s) for s in (1, 2)]
    datas = [bytearray(Codec("int4", compress="zstd").encode(a)) for a in arrs]
    out = kvcuda.BlockLoader().assemble([EncodedBlock(d) for d in datas])
    np.testing.assert_array_equal(out.cpu().numpy(), np.concatenate([Codec.decode(d) for d in datas], axis=3))


def test_store_writes_identical_files_with_gpu_codec(tmp_path):
    from kvwear.blocks import block_hashes
    from kvwear.policy import RequestContext, make_policy
    from kvwear.store import TieredKVStore

    def run(device, root):
        pol = make_policy("spill_int8")
        pol.codec.to_device(device)
        st = TieredKVStore(pol, root, ram_bytes=2**20, ssd_bytes=2**30, async_writes=False,
                           read_encoded=(device == "cuda"))
        ids = list(range(64 * 12))
        hs = block_hashes(ids, 64)
        st.begin_request(RequestContext(0, 0, 1, False), hs)
        for i, h in enumerate(hs):
            st.put(h, _block((2, 2, 4, 64, 64), seed=i))
        got = st.lookup_prefix(hs)                    # some come back from the SSD
        st.close()
        files = {p.name: p.read_bytes() for p in root.rglob("*.kvb")}
        return st.summary(), files, got

    s_cpu, f_cpu, got_cpu = run("cpu", tmp_path / "cpu")
    s_gpu, f_gpu, got_gpu = run("cuda", tmp_path / "gpu")
    assert f_cpu and f_cpu == f_gpu
    assert s_cpu["app_bytes_written"] == s_gpu["app_bytes_written"]
    assert s_cpu["ssd_hits"] == s_gpu["ssd_hits"] > 0
    assert any(isinstance(b, EncodedBlock) for b in got_gpu)
    for a, b in zip(got_cpu, got_gpu):
        b = b if isinstance(b, np.ndarray) else b.to_numpy()
        np.testing.assert_array_equal(a, b)


def test_runner_sim_engine_same_writes_with_gpu_codec(tmp_path):
    from kvwear.runner import main
    import json, io, contextlib

    def run(dev):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["--config", "configs/smoke.yaml", "--engine", "sim", "--policy", "spill_int4",
                  "--run-id", f"cuda-{dev}", "--quiet",
                  "--set", f"results_dir={tmp_path}", "--set", f"store.offload_dir={tmp_path}/off_{dev}",
                  "--set", f"store.codec_device={dev}", "--set", "engine.sim_decode_s_per_token=0",
                  "--set", "engine.sim_prefill_s_per_token=0"])
        line = [l for l in buf.getvalue().splitlines() if l.startswith("KVWEAR_RESULT")][0]
        return json.loads(line.split(" ", 1)[1])

    a, b = run("cpu"), run("cuda")
    assert a["app_bytes_written"] == b["app_bytes_written"] > 0
    assert b["codec_device"] == "cuda"


@pytest.mark.parametrize("hq,hkv,d", [(4, 4, 64), (14, 2, 64), (8, 2, 128), (16, 2, 64), (12, 4, 128)])
@pytest.mark.parametrize("tokens", [64, 200, 1000])
def test_paged_int8_attention_matches_reference(hq, hkv, d, tokens):
    from kvwear.cuda.attention import PagedInt8KV, dense_fp16_decode, reference
    rng = np.random.default_rng(tokens)
    L, Tb = 2, 64
    nblk = (tokens + Tb - 1) // Tb
    kv = rng.standard_normal((L, 2, hkv, nblk * Tb, d), dtype=np.float32).astype(np.float16)
    datas = [bytearray(Codec("int8").encode(np.ascontiguousarray(kv[:, :, :, i * Tb:(i + 1) * Tb])))
             for i in range(nblk)]
    paged = PagedInt8KV.from_payloads(datas)
    q = torch.randn(hq, d, dtype=torch.float16, device="cuda")
    deq = np.concatenate([Codec.decode(x) for x in datas], axis=3)   # what the kernel "sees"
    for layer in range(L):
        Kd = torch.from_numpy(np.ascontiguousarray(deq[layer, 0, :, :tokens])).cuda()
        Vd = torch.from_numpy(np.ascontiguousarray(deq[layer, 1, :, :tokens])).cuda()
        ref = reference(q, Kd, Vd)
        for variant in (1, 2, 3):
            out = paged.decode(q, layer, tokens=tokens, variant=variant).float()
            assert ((out - ref).norm() / ref.norm()).item() < 2e-3, variant
        dense = dense_fp16_decode(q, Kd, Vd).float()
        assert ((dense - ref).norm() / ref.norm()).item() < 2e-3


def test_int8_gpu_cache_append_and_gather_roundtrip():
    from kvwear.cuda.int8_cache import Int8GpuCache, cache_ext
    hkv, d, tmax = 2, 64, 300
    c = Int8GpuCache(layers=1, hkv=hkv, head_dim=d, max_tokens=tmax)
    ks, vs = [], []
    for t in (100, 1, 37):                                   # prefill, decode step, chunk
        k = torch.randn(1, hkv, t, d, dtype=torch.float16, device="cuda")
        v = torch.randn(1, hkv, t, d, dtype=torch.float16, device="cuda")
        c.update(k, v, 0)
        ks.append(k[0]); vs.append(v[0])
    T = c.get_seq_length(0)
    assert T == 138
    kd, vd = cache_ext().gather_int8(c.buf[0], hkv, T, d, tmax, c.g)
    kref = torch.cat(ks, 1).cpu().numpy()
    vref = torch.cat(vs, 1).cpu().numpy()
    # same quantization as the CPU codec, row by row
    for ref, got in ((kref, kd), (vref, vd)):
        exp = Codec.decode(Codec("int8").encode(ref.reshape(1, 1, hkv, T, d))).reshape(hkv, T, d)
        np.testing.assert_array_equal(got.cpu().numpy(), exp)


def test_int8_gpu_cache_decode_matches_reference():
    from kvwear.cuda.attention import reference
    from kvwear.cuda.int8_cache import Int8GpuCache, cache_ext
    hkv, hq, d, T = 2, 14, 64, 777
    c = Int8GpuCache(layers=2, hkv=hkv, head_dim=d, max_tokens=1024)
    for layer in range(2):
        c.update(torch.randn(1, hkv, T, d, dtype=torch.float16, device="cuda"),
                 torch.randn(1, hkv, T, d, dtype=torch.float16, device="cuda"), layer)
    q = torch.randn(hq, d, dtype=torch.float16, device="cuda")
    for layer in range(2):
        kd, vd = cache_ext().gather_int8(c.buf[layer], hkv, T, d, c.tmax, c.g)
        ref = reference(q, kd, vd)
        out = c.decode(q, layer, 1 / 8).float()
        assert ((out - ref).norm() / ref.norm()).item() < 2e-3
