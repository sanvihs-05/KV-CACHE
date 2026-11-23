"""The Nsight Compute report parser, on a synthetic `ncu --csv --page raw` export (no GPU needed)."""
from kvwear.cuda.ncu_report import attach_targets, parse_raw

RAW = '''"ID","Kernel Name","gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum","dram__throughput.avg.pct_of_peak_sustained_elapsed","sm__warps_active.avg.pct_of_peak_sustained_active","smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio","smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio","smsp__average_warps_issue_stalled_selected_per_issue_active.ratio"
"","","usecond","Mbyte","Kbyte","%","%","inst","inst","inst"
"0","quant_int8_subwarp<8>","70.5","12.58","6,291.5","93.1","88.0","9.5","0.2","1.0"
"1","paged_int8_partial_v5<64, 1, 4>","84.0","8.65","10","55","17","20.1","3.3","1.0"
"2","decode_combine","4.0","0.1","0.01","2","30","1","0","1"
'''


def test_parse_units_and_stalls():
    recs = parse_raw(RAW)
    assert len(recs) == 3
    a = recs[0]
    assert abs(a["time_us"] - 70.5) < 1e-9
    assert abs(a["dram_read"] - 12.58e6) < 1 and abs(a["dram_write"] - 6291.5e3) < 1   # "6,291.5" Kbyte
    assert a["stalls"][0][1] == "long_scoreboard"
    assert all(n != "selected" for _, n in a["stalls"])
    assert a["occ_theory"] is None                                                   # column absent


def test_combine_rows_fold_into_attention_target():
    recs = attach_targets(parse_raw(RAW), ["quantize int8 v4", "Qwen attention v5"])
    assert [r["target"] for r in recs] == ["quantize int8 v4", "Qwen attention v5"]
    assert abs(recs[1]["combine_us"] - 4.0) < 1e-9
