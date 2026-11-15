import json

import pandas as pd
import pytest

from kvwear import analysis, runner
from kvwear.config import load_config
from kvwear.engines.flexgen import build_command, parse_output

POLICIES = ["no_ssd", "write_through", "spill", "spill_int4", "reuse", "reuse_int4", "oracle"]


@pytest.fixture(scope="module")
def sim_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    out = {}
    for p in POLICIES:
        runner.main(["--config", "configs/smoke.yaml", "--engine", "sim", "--policy", p, "--run-id", p, "--quiet",
                     "--set", f"results_dir={tmp / 'results'}", "--set", f"store.offload_dir={tmp / 'off'}",
                     "--set", "engine.sim_prefill_s_per_token=0", "--set", "engine.sim_decode_s_per_token=0"])
        out[p] = json.loads((tmp / "results" / "runs" / p / "summary.json").read_text())
    return tmp, out


def test_sim_policies_order(sim_runs):
    _, r = sim_runs
    w = {p: r[p]["app_bytes_written"] for p in POLICIES}
    assert w["no_ssd"] == 0
    assert w["oracle"] < w["write_through"]
    assert w["spill"] <= w["write_through"]
    assert w["spill_int4"] < 0.35 * w["spill"]
    assert w["reuse_int4"] < w["reuse"]
    assert len({r[p]["prompt_tokens"] for p in POLICIES}) == 1          # identical trace for all
    # an upper bound on writes saved must not recompute more than writing everything
    assert r["oracle"]["recomputed_tokens"] <= r["write_through"]["recomputed_tokens"]
    assert r["no_ssd"]["recomputed_tokens"] >= r["write_through"]["recomputed_tokens"]


def test_analysis_report(sim_runs, tmp_path):
    tmp, _ = sim_runs
    # app bytes: on Linux the device-wide diskstats counter also sees unrelated writes
    analysis.main(["report", "--runs", str(tmp / "results" / "runs"), "--experiments", str(tmp / "missing.csv"),
                   "--out", str(tmp_path), "--source", "app_bytes_written"])
    s = pd.read_csv(tmp_path / "summary.csv", index_col=0)
    assert s.loc["write_through", "writes_saved_pct"] == pytest.approx(0)
    assert s.loc["no_ssd", "writes_saved_pct"] == pytest.approx(100)
    assert (tmp_path / "tradeoff.png").exists() and (tmp_path / "writes_per_policy.png").exists()


def test_idle_subtraction():
    df = pd.DataFrame({
        "policy": ["idle", "idle", "write_through"], "window_s": [100, 100, 50],
        "smart_bytes_delta": [1e6, 3e6, 10e6], "linux_bytes_written": [0, 0, 8e6],
    })
    out, rate = analysis.subtract_idle(df)
    assert rate == pytest.approx(2e4)
    assert out["smart_llm_bytes"].iloc[0] == pytest.approx(9e6)
    assert out["virt_amplification"].iloc[0] == pytest.approx(9 / 8)


def test_kv_attribution_uses_no_ssd_control():
    df = pd.DataFrame({
        "policy": ["idle", "no_ssd", "no_ssd", "write_through"], "window_s": [100, 100, 100, 100],
        "smart_bytes_delta": [1e8, 2.1e9, 2.3e9, 6.2e9], "linux_bytes_written": [0, 0, 0, 2e9],
        "exit_code": [0, 0, 0, 0],
    })
    out, rate = analysis.subtract_idle(df)
    wt = out[out["policy"] == "write_through"].iloc[0]
    assert wt["smart_kv_bytes"] == pytest.approx(6.1e9 - 2.1e9)        # minus idle, minus control mean
    assert wt["virt_amplification"] == pytest.approx(2.0)
    assert analysis.pick_source(out) == "smart_kv_bytes"


def test_wear_projection_matches_project_doc_example():
    r = analysis.wear_projection(2e9, 50, 300)           # 2 GB/chat, 50 chats/day, 300 TBW
    assert r["gb_per_day"] == pytest.approx(100)
    assert r["years_to_tbw"] == pytest.approx(300e12 / (100e9 * 365))


def test_flexgen_command_and_parser():
    cfg = load_config("configs/flexgen.yaml")
    cmd = build_command(cfg, "/tmp/off", "/tmp/log")
    i = cmd.index("--percent")
    assert cmd[i + 1:i + 7] == ["100", "0", "0", "0", "100", "0"]
    text = ("model size: 2.443 GB\tcache size: 0.398 GB\thidden size (p): 0.008 GB\n"
            "peak gpu mem: 3.112 GB\tprojected: False\n"
            "prefill latency: 1.20 s\tprefill throughput: 1706.67 token/s\n"
            "decode latency: 9.80 s\tdecode throughput: 12.65 token/s\n"
            "total latency: 11.00 s\ttotal throughput: 11.64 token/s")
    m = parse_output(text)
    assert m["flexgen_decode_throughput"] == pytest.approx(12.65)
    assert m["flexgen_cache_size_gb"] == pytest.approx(0.398)


def test_config_override():
    cfg = load_config("configs/default.yaml",
                      ["store.ram_mb=256", "workload.turns={doc_qa: [1, 2], debug: [1, 1], general: [1, 1]}"])
    assert cfg.store.ram_mb == 256 and cfg.workload.turns["doc_qa"] == (1, 2)
    with pytest.raises(KeyError):
        load_config(None, ["store.nope=1"])
