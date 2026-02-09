"""音素マッチングフィルタのテスト。case_loader で読み込んだテキストファイルのケースを利用する。"""

import argparse
import difflib
import sys
from pathlib import Path

import jaconv

# スクリプトとして実行したときにプロジェクトルートを path に追加
if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from style_bert_vits2.nlp.japanese.g2p import g2p
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from tests.support.case_loader import load_cases


def get_phoneme_sequence(text: str) -> list[str]:
    """テキストを正規化し、g2p で音素列を取得する。"""
    norm_text = normalize_text(text)
    phones, _, _ = g2p(norm_text, use_jp_extra=True, raise_yomi_error=False)
    return phones


def phoneme_sequences_match(seq_a: list[str], seq_b: list[str]) -> bool:
    """2つの音素列が完全一致するかどうかを返す。長さと各要素が等しい場合に True。"""
    return len(seq_a) == len(seq_b) and all(a == b for a, b in zip(seq_a, seq_b))


def get_phoneme_mismatch_count(seq_kanji: list[str], seq_kata: list[str]) -> int:
    """difflib.SequenceMatcher の opcodes のうち、delete と insert の回数だけを数えて返す。"""
    matcher = difflib.SequenceMatcher(None, seq_kanji, seq_kata)
    count = 0
    previous_phoneme = None
    he_pending = False
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        seg_kanji = seq_kanji[i1:i2]
        seg_kata = seq_kata[j1:j2]
        
        if he_pending:
            if seg_kanji and seg_kanji[0] == "e":
                he_pending = False
            else:
                count += 1
                he_pending = False
        
        if tag == "replace":
            if previous_phoneme == "o" and seg_kanji and seg_kata and seg_kanji[-1] == "o" and seg_kata[-1] == "u":
                continue
            if previous_phoneme == "e" and seg_kanji and seg_kata and seg_kanji[-1] == "e" and seg_kata[-1] == "i":
                continue
            count += 1
        elif tag == "delete":
            if seg_kanji and seg_kanji[-1] == "h":
                he_pending = True
                continue
            count += 1
        elif tag == "insert":
            count += 1
            
        if seg_kanji:
            previous_phoneme = seg_kanji[-1]
    return count


def is_effective_match(seq_kanji: list[str], seq_kata: list[str]) -> bool:
    """単純完全一致、または mismatch_count が 0 なら一致とみなす。"""
    if phoneme_sequences_match(seq_kanji, seq_kata):
        return True
    return get_phoneme_mismatch_count(seq_kanji, seq_kata) == 0


def format_phoneme_diff(seq_kanji: list[str], seq_kata: list[str]) -> tuple[list[str], int]:
    """2つの音素列の差分を difflib.SequenceMatcher で取得し、表示用の行リストと mismatch_count を返す。"""
    matcher = difflib.SequenceMatcher(None, seq_kanji, seq_kata)
    lines: list[str] = []
    previous_phoneme = None
    mismatch_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        seg_kanji = seq_kanji[i1:i2]
        seg_kata = seq_kata[j1:j2]
        if tag == "equal":
            lines.append(f"  一致: {' '.join(seg_kanji)}")
        elif tag == "replace":
            if previous_phoneme == "o" and seg_kanji and seg_kata and seg_kanji[-1] == "o" and seg_kata[-1] == "u":
                continue
            if previous_phoneme == "e" and seg_kanji and seg_kata and seg_kanji[-1] == "e" and seg_kata[-1] == "i":
                continue
            mismatch_count += 1
            lines.append(f"  差異: {' '.join(seg_kanji)}  -->  {' '.join(seg_kata)}")
        elif tag == "delete":
            lines.append(f"  削除: {' '.join(seg_kanji)}")
            mismatch_count += 1
        elif tag == "insert":
            lines.append(f"  挿入: {' '.join(seg_kata)}")
            mismatch_count += 1
        if seg_kanji:
            previous_phoneme = seg_kanji[-1]
    return lines, mismatch_count


def filter_cases_by_phoneme_match(
    cases: list[tuple[str, str, str]],
    *,
    match: bool,
) -> list[tuple[str, str, str]]:
    """音素列の一致/不一致でケースをフィルタする（多段階: 完全一致 or mismatch_count==0 を一致とみなす）。

    Args:
        cases: (filename, kanji_kana, kata) のリスト。
        match: True のとき一致しているものだけを返す。False のとき一致していないものだけを返す。

    Returns:
        条件を満たすケースのリスト。
    """
    result: list[tuple[str, str, str]] = []
    for filename, kanji_kana, kata in cases:
        phones_kanji = get_phoneme_sequence(kanji_kana)
        phones_kata = get_phoneme_sequence(jaconv.kata2hira(kata))
        if is_effective_match(phones_kanji, phones_kata) is match:
            result.append((filename, kanji_kana, kata))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="テキストファイルからテストケースを読み込み、音素マッチングフィルタで検証する。"
    )
    parser.add_argument(
        "-t",
        "--transcript",
        type=Path,
        required=True,
        metavar="FILE",
        help="テストケースファイル（1行1件、filename:漢字かな交じり:カタカナ の形式）",
    )
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--match",
        action="store_true",
        help="音素列が一致しているケースだけを出力する",
    )
    filter_group.add_argument(
        "--mismatch",
        action="store_true",
        help="音素列が一致していないケースだけを出力する",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="音素列の差分を詳細に出力する（--match / --mismatch と併用可）",
    )
    args = parser.parse_args()

    path = args.transcript
    if not path.exists():
        sys.exit(f"ファイルが見つかりません: {path}")

    cases = load_cases(path)
    if not cases:
        sys.exit(f"有効なケースがありません: {path}")

    matched_count = 0
    mismatch_count = 0  
    if args.match:
        filtered_cases = filter_cases_by_phoneme_match(cases, match=True)
        matched_count = len(filtered_cases)
        mismatch_count = len(cases) - matched_count
    elif args.mismatch:
        filtered_cases = filter_cases_by_phoneme_match(cases, match=False)
        mismatch_count = len(filtered_cases)
        matched_count = len(cases) - mismatch_count
    else:
        filtered_cases = cases

    show_diff_only = args.mismatch
    if args.detail:
        for filename, kanji_kana, kata in filtered_cases:
            phones_kanji = get_phoneme_sequence(kanji_kana)
            phones_kata = get_phoneme_sequence(jaconv.kata2hira(kata))
            print(f"{filename}: {kanji_kana!r} / {kata!r}")
            if show_diff_only:
                lines, case_mismatch_count = format_phoneme_diff(phones_kanji, phones_kata)
                for line in lines:
                    print(line)
                print(f"  不一致数: {case_mismatch_count}")
            else:
                print(f"  kan: {phones_kanji}")
                print(f"  kat: {phones_kata}")
    
    print(f"合計ケース数: {len(cases)}")
    print(f"一致数: {matched_count}")
    print(f"不一致数: {mismatch_count}")


if __name__ == "__main__":
    main()
