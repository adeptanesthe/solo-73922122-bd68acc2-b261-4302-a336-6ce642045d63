"""固定的字符级词表，两个模型共用，无需任何下载。

词表布局：
  0 = <pad>
  1 = <unk>
  2 = <eos>
  3..97 = 可打印 ASCII 字符（空格 0x20 到 '~' 0x7E，共 95 个）
词表大小固定为 98。
"""

from __future__ import annotations

PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
_FIRST_CHAR_ID = 3
_CHARS = [chr(code) for code in range(0x20, 0x7F)]  # 95 个可打印 ASCII 字符
VOCAB_SIZE = _FIRST_CHAR_ID + len(_CHARS)  # 98
_CHAR_TO_ID = {ch: _FIRST_CHAR_ID + i for i, ch in enumerate(_CHARS)}

# id -> 展示用文本
_DISPLAY = {PAD_ID: "<pad>", UNK_ID: "<unk>", EOS_ID: "<eos>"}
for _i, _ch in enumerate(_CHARS):
    _DISPLAY[_FIRST_CHAR_ID + _i] = _ch


def encode(text: str) -> list[int]:
    """把文本编码为 token id 序列（不自动追加 EOS）。"""
    ids: list[int] = []
    for ch in text:
        ids.append(_CHAR_TO_ID.get(ch, UNK_ID))
    return ids


def decode(ids: list[int]) -> str:
    """把 token id 序列解码为文本，遇到 EOS 截断。"""
    out: list[str] = []
    for i in ids:
        if i == EOS_ID:
            break
        if i in (PAD_ID, UNK_ID):
            continue
        out.append(_DISPLAY.get(i, ""))
    return "".join(out)


def render_token(token_id: int) -> str:
    """单个 token 的展示文本（用于网页逐 token 着色）。"""
    if token_id == EOS_ID:
        return "<eos>"
    if token_id == PAD_ID:
        return "<pad>"
    if token_id == UNK_ID:
        return "<unk>"
    return _DISPLAY[token_id]
