from kvwear.blocks import block_hashes
from kvwear.config import WorkloadConfig
from kvwear.workload import FakeTokenizer, generate_workload, turn_gap_stats


def wl(**kw):
    return generate_workload(WorkloadConfig(**kw), FakeTokenizer())


def test_deterministic():
    a, b = wl(seed=1), wl(seed=1)
    assert [r.prompt_ids for r in a.requests] == [r.prompt_ids for r in b.requests]
    assert [r.prompt_ids for r in wl(seed=2).requests] != [r.prompt_ids for r in a.requests]


def test_every_turn_once_and_last_turn_flag():
    w = wl(num_conversations=12)
    for c in w.conversations:
        turns = [r for r in w.requests if r.conv_id == c.conv_id]
        assert [r.turn for r in turns] == list(range(1, c.num_turns + 1))
        assert [r.is_last_turn for r in turns] == [False] * (c.num_turns - 1) + [True]


def test_context_budget_respected():
    w = wl(max_context_tokens=900, doc_tokens=(300, 400))
    assert max(len(r.prompt_ids) for r in w.requests) <= 900


def test_shared_system_prompt_gives_shared_blocks():
    w = wl(num_conversations=8)
    firsts = {tuple(block_hashes(r.prompt_ids, 64)[:3]) for r in w.requests}
    assert len(firsts) == 1                      # 200-token system prompt = 3 identical leading blocks


def test_next_turn_extends_previous_prefix():
    w = wl(num_conversations=6, max_context_tokens=100_000)
    prev_by_conv = {}
    for r in w.requests:
        prev = prev_by_conv.get(r.conv_id)
        if prev is not None:
            assert r.prompt_ids[:len(prev)] == prev   # history = previous prompt + scripted reply
        prev_by_conv[r.conv_id] = r.prompt_ids


def test_gap_stats():
    counts, gaps = turn_gap_stats(wl(num_conversations=30))
    assert gaps and min(gaps) >= 1
    assert all(reached >= cont for reached, cont in counts.values())


def test_block_hash_prefix_property():
    ids = list(range(300))
    h1, h2 = block_hashes(ids, 64), block_hashes(ids[:200] + [7] * 100, 64)
    assert len(h1) == 4 and h1[:3] == h2[:3] and h1[3] != h2[3]
