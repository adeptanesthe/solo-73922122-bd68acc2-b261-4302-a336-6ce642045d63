"""关键功能测试：
  1. 带缓存与无缓存两条前向路径对同一序列给出逐位相同的 logits；
  2. 推测解码与无缓存贪心基线在多组提示词/k/max_new 下逐 token 完全一致；
  3. 目标前向次数 == 推测轮数，且严格少于基线（存在接受时）；
  4. KV 缓存长度不变量：目标每轮只吃新块，不重算已缓存前缀；
  5. EOS 截断、max_new 上限、平局取小 id 的行为正确；
  6. 预设提示词确实分别覆盖 reject 与 all_accepted。
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import tokenizer
from app.decoding import compare, greedy_baseline, speculative_decode
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
      - 后续轮目标前向输入恒为 1 个 pending + 草稿块，绝不包含已缓存前缀。
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

    ids2 = tokenizer.encode(PROMPTS[2])
    res2 = speculative_decode(DRAFT, TARGET, ids2, 2, 10)
    confirmed2 = len(ids2) + len(res2.new_tokens)
    assert DRAFT.cache_len == confirmed2
    assert TARGET.cache_len in (confirmed2 - 1, confirmed2)
    print("ok: KV 缓存不变量成立，已缓存前缀未被重算")


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
    from app.main import PRESET_ALL_ACCEPTED, PRESET_REJECT

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
    test_max_new_hard_cap()
    test_eos_handling()
    test_round_traces_contain_both_decisions()
    print("\n全部测试通过 ✔")
