import re

from style_bert_vits2.nlp.japanese.g2p import g2p
from style_bert_vits2.nlp.japanese.mora_list import (
    CONSONANTS,
    MORA_KATA_TO_MORA_PHONEMES,
    MORA_PHONEMES_TO_MORA_KATA,
)
from style_bert_vits2.nlp.symbols import PUNCTUATIONS


def g2kata_tone(norm_text: str) -> list[tuple[str, int]]:
    """
    テキストからカタカナとアクセントのペアのリストを返す。
    推論時のみに使われる関数のため、常に `raise_yomi_error=False` を指定して g2p() を呼ぶ仕様になっている。

    Args:
        norm_text: 正規化されたテキスト。

    Returns:
        カタカナと音高のリスト。
    """

    phones, tones, _ = g2p(norm_text, use_jp_extra=True, raise_yomi_error=False)
    return phone_tone2kata_tone(list(zip(phones, tones)))


def g2kata_tone_safe(
    text: str,
    *,
    max_unit_chars: int = 300,
) -> list[tuple[str, int]]:
    """
    長文でも worker を落としにくいように、内部で分割して g2kata_tone を実行する。

    - 改行で分割し、さらに長い行は句読点・固定長で分割する
    - 分割境界には「、」(tone=0) を挿入する（アクセント指定 UI 上の区切り用）
    """

    def split_long_segment(seg: str) -> list[str]:
        seg = seg.strip()
        if not seg:
            return []
        if len(seg) <= max_unit_chars:
            return [seg]
        parts = re.split(r"([。！？!?…]+)", seg)
        merged: list[str] = []
        buf = ""
        for i in range(0, len(parts), 2):
            piece = parts[i]
            punct = parts[i + 1] if i + 1 < len(parts) else ""
            chunk = (piece + punct).strip()
            if not chunk:
                continue
            if not buf:
                buf = chunk
                continue
            if len(buf) + len(chunk) <= max_unit_chars:
                buf += chunk
            else:
                merged.append(buf)
                buf = chunk
        if buf:
            merged.append(buf)

        final: list[str] = []
        for m in merged:
            if len(m) <= max_unit_chars:
                final.append(m)
            else:
                for j in range(0, len(m), max_unit_chars):
                    s = m[j : j + max_unit_chars].strip()
                    if s:
                        final.append(s)
        return final

    lines = [ln for ln in text.split("\n") if ln.strip() != ""]
    result: list[tuple[str, int]] = []
    for li, line in enumerate(lines):
        chunks = split_long_segment(line)
        for ci, chunk in enumerate(chunks):
            if result:
                # 分割境界（改行・チャンク境界）を UI 上で分かるようにする
                result.append(("、", 0))
            result.extend(g2kata_tone(chunk))
        if li < len(lines) - 1 and result:
            # 行境界も区切る（チャンク境界で既に入っていれば重複するが UI 上の区切りとして許容）
            result.append(("、", 0))
    return result


def phone_tone2kata_tone(phone_tone: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    phone_tone の phone 部分をカタカナに変換する。ただし最初と最後の ("_", 0) は無視する。

    Args:
        phone_tone: 音素と音高のリスト。

    Returns:
        カタカナと音高のリスト。
    """

    phone_tone = phone_tone[1:]  # 最初の("_", 0)を無視
    phones = [phone for phone, _ in phone_tone]
    tones = [tone for _, tone in phone_tone]
    result: list[tuple[str, int]] = []
    current_mora = ""
    for phone, next_phone, tone, next_tone in zip(phones, phones[1:], tones, tones[1:]):
        # zip の関係で最後の ("_", 0) は無視されている
        if phone in PUNCTUATIONS:
            result.append((phone, tone))
            continue
        if phone in CONSONANTS:  # n以外の子音の場合
            assert current_mora == "", f"Unexpected {phone} after {current_mora}"
            assert tone == next_tone, f"Unexpected {phone} tone {tone} != {next_tone}"
            current_mora = phone
        else:
            # phoneが母音もしくは「N」
            current_mora += phone
            result.append((MORA_PHONEMES_TO_MORA_KATA[current_mora], tone))
            current_mora = ""

    return result


def kata_tone2phone_tone(kata_tone: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    `phone_tone2kata_tone()` の逆の変換を行う。

    Args:
        kata_tone: カタカナと音高のリスト。

    Returns:
        音素と音高のリスト。
    """

    result: list[tuple[str, int]] = [("_", 0)]
    for mora, tone in kata_tone:
        if mora in PUNCTUATIONS:
            result.append((mora, tone))
        else:
            consonant, vowel = MORA_KATA_TO_MORA_PHONEMES[mora]
            if consonant is None:
                result.append((vowel, tone))
            else:
                result.append((consonant, tone))
                result.append((vowel, tone))
    result.append(("_", 0))

    return result
