"""FastAPI 后端：推测解码 vs 无缓存贪心基线的对照服务。"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import tokenizer
from .decoding import compare
from .model import get_models

app = FastAPI(title="Speculative Decoding Verifier")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")

# 模型是进程内单例，KV 缓存与前向计数是请求级可变状态；同步路由在
# 线程池中并发执行，必须串行化整次解码，否则请求之间会互相污染缓存。
_DECODE_LOCK = threading.Lock()

# 固定预设：经实测与测试套件锁定，分别稳定触发"拒绝"与"整块接受"。
# 拒绝预设：k=3 第 2 轮为"部分接受后拒绝"（草稿 [空格,K,K]，只接受 1 个），
# 用于验证被拒 token 不会残留、下一轮草稿仍基于完整确认前缀；
# k=4 第 2 轮全部被拒、第 3 轮即恢复整块接受。
# 整块接受预设：k=4 时各轮草稿 token 全部通过，展示目标前向节省的上限。
PRESET_REJECT = "python java rust go c plus plus ruby swift scala "
PRESET_ALL_ACCEPTED = "the quick brown fox jumps over the lazy dog while "


class DecodeRequest(BaseModel):
    prompt: str = Field(default="", max_length=200)
    k: int = Field(default=3, ge=1, le=4)
    max_new: int = Field(default=32, ge=1, le=32)


def _token_view(ids: list[int]) -> list[dict]:
    return [{"id": int(t), "text": tokenizer.render_token(int(t))} for t in ids]


def _serialize(cmp_result, prompt_text: str) -> dict:
    base, spec = cmp_result.baseline, cmp_result.speculative
    return {
        "prompt": prompt_text,
        "prompt_ids": base.prompt_ids,
        "match": cmp_result.match,
        "status": "PASS" if cmp_result.match else "FAIL",
        "first_mismatch_index": cmp_result.first_mismatch_index,
        "baseline": {
            "new_tokens": _token_view(base.new_tokens),
            "text": tokenizer.decode(base.new_tokens),
            "target_forward_count": base.target_forward_count,
            "stopped_by_eos": base.stopped_by_eos,
        },
        "speculative": {
            "new_tokens": _token_view(spec.new_tokens),
            "text": tokenizer.decode(spec.new_tokens),
            "target_forward_count": spec.target_forward_count,
            "draft_forward_count": spec.draft_forward_count,
            "stopped_by_eos": spec.stopped_by_eos,
            "rounds": [asdict(r) | {
                "draft_tokens_view": _token_view(r.draft_tokens),
                "target_predictions_view": _token_view(r.target_predictions),
                "target_tokens_view": _token_view(r.target_tokens),
            } for r in spec.rounds],
        },
        "savings": {
            "baseline_forwards": base.target_forward_count,
            "spec_forwards": spec.target_forward_count,
            "saved": base.target_forward_count - spec.target_forward_count,
        },
    }


@app.get("/api/health")
def health() -> dict:
    draft, target = get_models()
    return {
        "ok": True,
        "vocab_size": tokenizer.VOCAB_SIZE,
        "target_layers": target.n_layers,
        "draft_layers": draft.n_layers,
    }


@app.post("/api/decode")
def decode(req: DecodeRequest) -> JSONResponse:
    prompt_ids = tokenizer.encode(req.prompt)
    if not prompt_ids:
        return JSONResponse(
            status_code=400,
            content={"error": "提示词编码后为空，请输入可打印 ASCII 字符。"},
        )
    draft, target = get_models()
    with _DECODE_LOCK:
        result = compare(draft, target, prompt_ids, req.k, req.max_new)
    return JSONResponse(_serialize(result, req.prompt))


@app.get("/api/presets")
def presets() -> dict:
    """返回固定的两条预设（由测试套件锁定其判定类型，避免请求时全量扫描）。"""
    return {"reject": PRESET_REJECT, "all_accepted": PRESET_ALL_ACCEPTED}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
