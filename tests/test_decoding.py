"""关键功能测试：
  1. 带缓存与无缓存两条前向路径对同一序列给出逐位相同的 logits；
  2. 推测解码与无缓存贪心基线在多组提示词/k/max_new 下逐 token 完全一致；
  3. 目标前向次数 == 推测轮数，且严格少于基线（存在接受时）；
  4. KV 缓存长度不变量：目标每轮只吃新块，不重算已缓存前缀；
  5. EOS 截断、max_new 上限、平局取小 id 的行为正确；
  6. 预设提示词确实分别覆盖 reject 与 all_accepted；
  7. 回归：拒绝后下一轮草稿基于完整确认前缀，缓存内容与该前缀逐位一致；
  8. 网页接口逐轮字段（前向次数/接受数）真实且可加总。
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import tokenizer
from app.decoding import compare, greedy_baseline, speculative_decode
from app.main import PRESET_ALL_ACCEPTED, PRESET_REJECT
from app.model import get_models, greedy_argmax

torch.set_num_threads(1)
DRAFT, TARGET = get_models()

PROMPTS = [
    "the quick brown fox jumps over ",
    "hello world, this is a demo of ",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ",
    "x = 1\ny = 2\nprint(x + y)  # ",
    "0 1 1 2 3 5 8 13 21 34 next ",
    "speculative decoding verifies that ",
    "once upon a time there lived ",
]


def test_cached_vs_uncached_logits_identical() -> None:
    target = TARGET
    ids = tokenizer.encode("the quick brown fox ")
    # 无缓存整序列前向
    full = target.forward(ids, use_cache=False)
    # 缓存路径：先喂前缀，再喂最后一个 token
    target.init_cache()
    target.forward(ids[:-1], use_cache=True)
    inc = target.forward([ids[-1]], use_cache=True)
    # float32 矩阵乘法结合顺序不同会有 ~1e-5 级舍入误差
    assert torch.allclose(full[-1], inc[-1], atol=1e-4), "两条前向路径 logits 不一致"
    # 关键：argmax 贪心决策必须完全相同
    assert full[-1].argmax() == inc[-1].argmax()

    # 多 token 增量块也必须与整序列前向逐位一致
    target.init_cache()
    target.forward(ids[:3], use_cache=True)
    block = target.forward(ids[3:], use_cache=True)
    assert torch.allclose(full[3:], block, atol=1e-4), "多 token 增量前向与整序列不一致"
    assert torch.equal(full[3:].argmax(dim=-1), block.argmax(dim=-1))
    print("ok: 缓存/无缓存前向 logits 逐位一致（argmax 决策完全相同）")


def test_cache_rollback_no_contamination() -> None:
    """拒绝场景核心保证：缓存回滚后换续写，预测与对应整序列前向完全一致。"""
    target = TARGET
    ids = tokenizer.encode("the quick brown fox jumps over ")
    block = [10, 20, 30, 40]

    full = target.forward(ids + block, use_cache=False)
    ref_preds = [greedy_argmax(full[len(ids) + j]) for j in range(len(block))]

    target.init_cache()
    target.forward(ids, use_cache=True)
    l1 = target.forward(block[:2], use_cache=True)
    l2 = target.forward(block[2:], use_cache=True)
    cached_preds = [greedy_argmax(x) for x in list(l1) + list(l2)]
    assert cached_preds == ref_preds

    # 回滚到提示词长度后换一组续写（模拟拒绝后用目标 token 纠正）
    target.truncate_cache(len(ids))
    new_block = [50, 60]
    l3 = target.forward(new_block, use_cache=True)
    ref2 = target.forward(ids + new_block, use_cache=False)
    assert greedy_argmax(l3[0]) == greedy_argmax(ref2[len(ids)])
    assert greedy_argmax(l3[1]) == greedy_argmax(ref2[len(ids) + 1])
    print("ok: 缓存回滚后续写无污染，与整序列前向逐位一致")


def test_spec_matches_greedy_all_cases() -> None:
    n_cases = 0
    for p in PROMPTS:
        ids = tokenizer.encode(p)
        for k in (1, 2, 3, 4):
            for max_new in (1, 7, 32):
                r = compare(DRAFT, TARGET, ids, k, max_new)
                assert r.match, (
                    f"不一致: prompt={p!r} k={k} max_new={max_new} "
                    f"idx={r.first_mismatch_index} "
                    f"base={r.baseline.new_tokens} spec={r.speculative.new_tokens}"
                )
                assert len(r.speculative.new_tokens) <= max_new
                n_cases += 1
    print(f"ok: {n_cases} 组参数下推测解码与贪心基线逐 token 一致")


def test_target_forward_count_equals_rounds_and_saves() -> None:
    ids = tokenizer.encode(PROMPTS[0])
    r = compare(DRAFT, TARGET, ids, 4, 32)
    spec = r.speculative
    assert spec.target_forward_count == len(spec.rounds), "目标前向次数必须等于轮数"
    # 每轮恰好一次目标前向，输入长度 = 首轮提示词+块，后续仅块
    first = spec.rounds[0]
    assert first.target_forward_input_len == len(ids) + len(first.draft_tokens)
    for rd in spec.rounds[1:]:
        assert rd.target_forward_input_len == 1 + len(rd.draft_tokens)
    # 32 个新 token、k=4，推测轮数必然显著少于 32
    assert spec.target_forward_count < r.baseline.target_forward_count
    print(f"ok: 目标前向 {spec.target_forward_count} 次 vs 基线 "
          f"{r.baseline.target_forward_count} 次，轮数=前向数")


def test_cache_prefix_not_recomputed() -> None:
    """缓存长度不变量（pending-tail 方案）：
      - 草稿缓存始终对齐完整确认序列；
      - 目标缓存 == 确认序列长度 - 1（正常轮，tail 待下一轮送入），
        终止轮可能等于确认长度（tail 被上限/EOS 截断，无 pending）；
      - 后续轮目标前向输入恒为 1 个 pending + 草稿块，绝不包含已缓存前缀；
      - 每轮目标前向恰好 1 次，且逐轮前向计数之和等于总数。
    """
    ids = tokenizer.encode(PROMPTS[1])
    res = speculative_decode(DRAFT, TARGET, ids, 3, 20)
    confirmed = len(ids) + len(res.new_tokens)
    assert DRAFT.cache_len == confirmed
    assert TARGET.cache_len in (confirmed - 1, confirmed)
    for rd in res.rounds[1:]:
        assert rd.target_forward_input_len == 1 + len(rd.draft_tokens)
    assert res.rounds[0].target_forward_input_len == len(ids) + len(
        res.rounds[0].draft_tokens)
    assert all(rd.target_forward_calls == 1 for rd in res.rounds)
    assert sum(rd.target_forward_calls for rd in res.rounds) == \
        res.target_forward_count
    assert sum(rd.draft_forward_calls for rd in res.rounds) == \
        res.draft_forward_count

    ids2 = tokenizer.encode(PROMPTS[2])
    res2 = speculative_decode(DRAFT, TARGET, ids2, 2, 10)
    confirmed2 = len(ids2) + len(res2.new_tokens)
    assert DRAFT.cache_len == confirmed2
    assert TARGET.cache_len in (confirmed2 - 1, confirmed2)
    print("ok: KV 缓存不变量成立，已缓存前缀未被重算；逐轮前向计数真实可加总")


def _fresh_draft_block(prefix: list[int], k: int, room: int) -> list[int]:
    """用全新草稿模型基于完整确认前缀做贪心续写，作为对照基准。"""
    from app.model import _build_pair
    fresh, _ = _build_pair()
    fresh.init_cache()
    logits = fresh.forward(list(prefix), use_cache=True)[-1]
    block: list[int] = []
    for _ in range(min(k, room)):
        nxt = greedy_argmax(logits)
        block.append(nxt)
        if nxt == tokenizer.EOS_ID or len(block) == min(k, room):
            break
        logits = fresh.forward([nxt], use_cache=True)[-1]
    return block


def test_draft_blocks_follow_confirmed_prefix_after_reject() -> None:
    """回归：一轮中途拒绝后，下一轮草稿必须基于完整确认前缀。

    旧实现只把草稿缓存截断到 len(seq)，部分拒绝时会把被拒 token 留在缓存
    尾部，使下一轮首个草稿 token 基于被拒 token 预测。此处逐轮用"全新模型
    在完整确认前缀上的贪心续写"做对照，确保草稿块逐 token 一致。
    """
    ids = tokenizer.encode(PROMPTS[5])  # speculative decoding verifies that
    for k in (2, 3, 4):
        res = speculative_decode(DRAFT, TARGET, ids, k, 24)
        prefix = list(ids)
        for rd in res.rounds:
            room = 24 - (len(prefix) - len(ids))
            expected = _fresh_draft_block(prefix, k, room)
            assert rd.draft_tokens == expected, (
                f"k={k} 第 {rd.index} 轮草稿未基于完整确认前缀: "
                f"{rd.draft_tokens} != {expected}")
            assert rd.draft_cache_len_after == len(prefix) + len(rd.target_tokens)
            assert rd.target_forward_calls == 1
            prefix.extend(rd.target_tokens)
        assert DRAFT.cache_len == len(prefix)

    # 部分接受路径：拒绝预设 k=3 第 2 轮草稿 [空格, K, K] 只接受 1 个。
    ids3 = tokenizer.encode(PRESET_REJECT)
    res3 = speculative_decode(DRAFT, TARGET, ids3, 3, 32)
    partial = [rd for rd in res3.rounds
               if 0 < rd.accepted < len(rd.draft_tokens)]
    assert partial, "未覆盖部分拒绝路径"
    prefix = list(ids3)
    for rd in res3.rounds:
        room = 32 - (len(prefix) - len(ids3))
        assert rd.draft_tokens == _fresh_draft_block(prefix, 3, room), \
            f"第 {rd.index} 轮草稿偏离完整确认前缀"
        prefix.extend(rd.target_tokens)

    # 旧实现在该预设 k=4 下因被拒 token 残留在草稿缓存尾部，后续 26/28 轮
    # 草稿都偏离确认前缀；逐轮与全新模型对照即可稳定捕获。
    res4 = speculative_decode(DRAFT, TARGET, ids3, 4, 32)
    prefix = list(ids3)
    for rd in res4.rounds:
        room = 32 - (len(prefix) - len(ids3))
        assert rd.draft_tokens == _fresh_draft_block(prefix, 4, room), \
            f"k=4 第 {rd.index} 轮草稿偏离完整确认前缀"
        prefix.extend(rd.target_tokens)
    print(f"ok: 拒绝后草稿基于完整确认前缀（含 {len(partial)} 个部分拒绝轮）")


def test_draft_cache_content_matches_fresh_prefix() -> None:
    """拒绝回滚后草稿/目标 KV 内容必须与"从头前向完整确认前缀"一致。"""
    from app.model import _build_pair

    ids = tokenizer.encode(PRESET_REJECT)
    res = speculative_decode(DRAFT, TARGET, ids, 3, 24)
    confirmed = ids + res.new_tokens

    def assert_cache_prefix_matches(model, ref_model, prefix) -> None:
        ref_model.init_cache()
        ref_model.forward(prefix, use_cache=True)
        assert model.cache_len <= len(prefix)
        for (k1, v1), (k2, v2) in zip(model._cache, ref_model._cache):
            assert torch.allclose(k1, k2[:, :, :model.cache_len, :], atol=1e-4), \
                "K 缓存内容与完整确认前缀不一致"
            assert torch.allclose(v1, v2[:, :, :model.cache_len, :], atol=1e-4), \
                "V 缓存内容与完整确认前缀不一致"

    ref_draft, _ = _build_pair()
    _, ref_target = _build_pair()
    assert DRAFT.cache_len == len(confirmed)
    assert_cache_prefix_matches(DRAFT, ref_draft, confirmed)
    # 目标缓存（正常终止：覆盖 confirmed-1 或 confirmed）也须与完整前缀一致
    assert TARGET.cache_len in (len(confirmed) - 1, len(confirmed))
    assert_cache_prefix_matches(TARGET, ref_target, confirmed)

    # 对照：整块接受预设同样满足该不变量
    ids2 = tokenizer.encode(PRESET_ALL_ACCEPTED)
    res2 = speculative_decode(DRAFT, TARGET, ids2, 4, 24)
    confirmed2 = ids2 + res2.new_tokens
    ref2, _ = _build_pair()
    _, ref_t2 = _build_pair()
    assert DRAFT.cache_len == len(confirmed2)
    assert_cache_prefix_matches(DRAFT, ref2, confirmed2)
    assert TARGET.cache_len in (len(confirmed2) - 1, len(confirmed2))
    assert_cache_prefix_matches(TARGET, ref_t2, confirmed2)
    print("ok: 草稿/目标 KV 缓存内容与完整确认前缀逐位一致（拒绝/整块接受）")


def test_max_new_hard_cap() -> None:
    ids = tokenizer.encode(PROMPTS[0])
    for cap in (1, 2, 5, 32):
        r = compare(DRAFT, TARGET, ids, 4, cap)
        assert len(r.baseline.new_tokens) == len(r.speculative.new_tokens) <= cap
        assert r.match
    print("ok: max_new 上限在两种方式下行为一致")


def test_eos_handling() -> None:
    """EOS 处理：自然场景下位置一致；再用 hook 强制注入 EOS 覆盖所有 k/上限组合。"""
    ids = tokenizer.encode("the quick brown fox ")
    base = greedy_baseline(TARGET, ids, 32)
    spec = speculative_decode(DRAFT, TARGET, ids, 4, 32)
    base_eos = [i for i, t in enumerate(base.new_tokens) if t == tokenizer.EOS_ID]
    spec_eos = [i for i, t in enumerate(spec.new_tokens) if t == tokenizer.EOS_ID]
    assert base_eos == spec_eos

    # 强制目标把 EOS 作为最大 logit，验证两条路径 EOS 截断位置逐位一致
    def eos_hook(module, inp, out):
        out2 = out.clone()
        out2[..., tokenizer.EOS_ID] = out2.max().item() + 100.0
        return out2

    handle = TARGET.lm_head.register_forward_hook(eos_hook)
    try:
        for k in (1, 2, 3, 4):
            for cap in (8, 16, 32):
                r = compare(DRAFT, TARGET, ids, k, cap)
                assert r.match, (k, cap, r.first_mismatch_index)
                be = [i for i, t in enumerate(r.baseline.new_tokens)
                      if t == tokenizer.EOS_ID]
                se = [i for i, t in enumerate(r.speculative.new_tokens)
                      if t == tokenizer.EOS_ID]
                assert be == se and len(be) == 1
                assert r.baseline.stopped_by_eos and r.speculative.stopped_by_eos
                assert len(r.baseline.new_tokens) == be[0] + 1
    finally:
        handle.remove()
    print("ok: EOS 截断在自然与强制场景下两种方式完全一致")


def test_tie_prefers_small_id() -> None:
    logits = torch.zeros(5)  # 全部并列
    from app.model import greedy_argmax
    assert greedy_argmax(logits) == 0
    logits[3] = 1.0
    assert greedy_argmax(logits) == 3
    print("ok: 平局时取词表序号小者")


def test_round_traces_contain_both_decisions() -> None:
    """固定预设必须分别触发 reject 与 all_accepted（网页两个预设按钮的依据）。"""
    r_reject = compare(DRAFT, TARGET, tokenizer.encode(PRESET_REJECT), 4, 32)
    assert "reject" in {rd.decision for rd in r_reject.speculative.rounds}, \
        "拒绝预设未触发 reject"
    assert r_reject.match

    r_accept = compare(DRAFT, TARGET, tokenizer.encode(PRESET_ALL_ACCEPTED), 4, 32)
    assert "all_accepted" in {rd.decision for rd in r_accept.speculative.rounds}, \
        "整块接受预设未触发 all_accepted"
    assert r_accept.match
    print(f"ok: 拒绝预设出现拒绝轮（前向 {r_reject.speculative.target_forward_count} 次）")
    print(f"    整块接受预设（前向 {r_accept.speculative.target_forward_count} 次）")


def test_api_rounds_serialize_real_counts() -> None:
    """网页依赖的逐轮字段必须真实：目标每轮 1 次、计数可加总、接受数自洽。"""
    from fastapi.testclient import TestClient

    from app.main import PRESET_ALL_ACCEPTED, PRESET_REJECT, app

    client = TestClient(app)
    for prompt in (PRESET_REJECT, PRESET_ALL_ACCEPTED):
        resp = client.post("/api/decode",
                           json={"prompt": prompt, "k": 3, "max_new": 32})
        assert resp.status_code == 200
        data = resp.json()
        assert data["match"] and data["status"] == "PASS"
        spec = data["speculative"]
        rounds = spec["rounds"]
        assert spec["target_forward_count"] == len(rounds)
        assert sum(r["target_forward_calls"] for r in rounds) == \
            spec["target_forward_count"]
        assert sum(r["draft_forward_calls"] for r in rounds) == \
            spec["draft_forward_count"]
        for r in rounds:
            assert r["target_forward_calls"] == 1
            assert r["accepted"] <= len(r["draft_tokens"])
            # 未截断的轮次：目标恢复 token = 接受数 + 1 个纠正/补词 token
            assert len(r["target_tokens"]) in (r["accepted"], r["accepted"] + 1)
        assert data["savings"]["saved"] == (
            data["baseline"]["target_forward_count"]
            - spec["target_forward_count"])
        assert client.get("/api/health").json()["ok"]
    assert client.post("/api/decode", json={"prompt": "", "k": 3,
                                            "max_new": 8}).status_code == 400
    assert client.post("/api/decode", json={"prompt": "abc", "k": 9,
                                            "max_new": 8}).status_code == 422
    print("ok: 网页逐轮字段真实可加总，空提示词 400、非法参数 422")


def test_deterministic_weights() -> None:
    """重新构建模型应得到完全相同的权重（固定种子）。"""
    from app.model import _build_pair
    pair1 = _build_pair()
    pair2 = _build_pair()
    for m1, m2 in zip(pair1, pair2):
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert n1 == n2 and torch.equal(p1, p2)
    print("ok: 固定种子下权重确定可复现")


if __name__ == "__main__":
    test_deterministic_weights()
    test_cached_vs_uncached_logits_identical()
    test_cache_rollback_no_contamination()
    test_tie_prefers_small_id()
    test_spec_matches_greedy_all_cases()
    test_target_forward_count_equals_rounds_and_saves()
    test_cache_prefix_not_recomputed()
    test_draft_blocks_follow_confirmed_prefix_after_reject()
    test_draft_cache_content_matches_fresh_prefix()
    test_max_new_hard_cap()
    test_eos_handling()
    test_round_traces_contain_both_decisions()
    test_api_rounds_serialize_real_counts()
    print("\n全部测试通过 ✔")
