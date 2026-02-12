#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音声データと書き起こしテキストをStyle-Bert-VITS2のフォーマットに変換するスクリプト

単一話者（従来）:
    モデルディレクトリ直下に transcript_utf8.txt と wav/ を置く。
    python convert_data_format.py --model_name my_model --speaker_name spk1 --language JP

複数話者:
    モデルディレクトリ直下に話者名のサブディレクトリを置き、各サブディレクトリに
    書き起こしファイルと wav/ を置く。
    Data/{model_name}/{speaker_name}/transcript_utf8.txt
    Data/{model_name}/{speaker_name}/wav/
    python convert_data_format.py --model_name my_model --language JP

引数:
    --input_dirpath: 入力用Data相当ディレクトリ（デフォルト: スクリプト所在/Data）
    --model_name: モデル名（Data/配下のディレクトリ名）
    --speaker_name: 話者名（単一話者時のみ。デフォルト: モデル名）
    --language: 言語ID（JP/EN/ZH、デフォルト: JP）
    --filter-phoneme-match: 音素マッチするケースのみ変換する（JP のみ）

出力は常に Data/raw と Data/esd.list に固定される。
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def parse_transcript(transcript_path: Path) -> Dict[str, str]:
    """
    書き起こしファイルを解析して、ファイル名（stem）と書き起こしテキストのマッピングを返す。

    期待フォーマット: ファイル名:書き起こしテキスト
    ファイル名は「file01」「file01.wav」のどちらでも可。拡張子は除去して stem で照合する。

    Args:
        transcript_path: 書き起こしファイルのパス

    Returns:
        ファイル名（拡張子なし）をキー、書き起こしテキストを値とする辞書
    """
    transcript_map: Dict[str, str] = {}

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if ":" in line:
                parts = line.split(":", 2)
                filename_base = Path(parts[0].strip()).stem
                # 2列(filename:text)でも3列(filename:漢字かな:カタカナ)でも漢字かな交じり文のみ使う
                text = parts[1].strip()
                transcript_map[filename_base] = text

    return transcript_map


def _derive_kata_ja(text: str) -> str:
    """日本語テキストからカタカナ読みを取得する（正規化＋g2p）。"""
    from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata
    from style_bert_vits2.nlp.japanese.normalizer import normalize_text

    norm_text = normalize_text(text)
    _, sep_kata = text_to_sep_kata(norm_text, raise_yomi_error=False)
    return "".join(sep_kata)


def parse_transcript_to_cases(transcript_path: Path) -> List[Tuple[str, str, str]]:
    """
    書き起こしファイルを (filename_base, kanji_kana, kata) のリストとして読み込む。
    2列形式（filename:text）の場合は kata を g2p で導出する。
    3列形式（filename:kanji_kana:kata）の場合はそのまま利用する。
    """
    cases: List[Tuple[str, str, str]] = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":", 2)
            if len(parts) < 2:
                continue
            filename_base = Path(parts[0].strip()).stem
            if len(parts) == 2:
                kanji_kana = parts[1].strip()
                kata = _derive_kata_ja(kanji_kana)
            else:
                kanji_kana = parts[1].strip()
                kata = parts[2].strip()
            cases.append((filename_base, kanji_kana, kata))
    return cases


def _warn_missing_audio(
    speaker_label: str,
    missing: List[str],
    raw_audio_map: Dict[str, Path],
) -> None:
    """音声が見つからない件をまとめて表示し、想定原因を示す。"""
    if not missing:
        return
    n = len(missing)
    examples = missing[:5] if n > 5 else missing
    found = list(raw_audio_map.keys())[:5]
    print(f"  警告 [{speaker_label}]: 音声が見つからない書き起こし: {n}件 (例: {', '.join(examples)})")
    print("    想定原因: (1) wav/に該当ファイルが無い (2) 書き起こしと実ファイル名の桁数違い（例: -01 と -0001）")
    if found:
        print(f"    参考: 実在するファイル名の例: {', '.join(found)}")


def get_audio_files(audio_dir: Path) -> Dict[str, Path]:
    """
    音声ファイルのマッピングを取得
    
    Args:
        audio_dir: 音声ファイルが格納されているディレクトリ
        
    Returns:
        ファイル名（拡張子なし）をキー、ファイルパスを値とする辞書
    """
    audio_map = {}
    
    # サポートする音声ファイルの拡張子
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus'}
    
    for audio_file in audio_dir.rglob('*'):
        if audio_file.suffix.lower() in audio_extensions:
            filename_base = audio_file.stem
            audio_map[filename_base] = audio_file
    return audio_map

def get_speaker_subdirs(
    model_dir: Path,
    transcript_file: str,
    audio_dir: str,
) -> List[str]:
    """
    モデルディレクトリ直下の、書き起こしと音声ディレクトリの両方を持つ
    話者名サブディレクトリの名前リストを返す。
    """
    speaker_dirs = []
    for path in model_dir.iterdir():
        if not path.is_dir():
            continue
        if (path / transcript_file).exists() and (path / audio_dir).is_dir():
            speaker_dirs.append(path.name)
    return sorted(speaker_dirs)


def write_esd_list(
    esd_list_path: Path,
    entries: List[Tuple[str, str, str]],
    language: str,
) -> None:
    """
    (path_str, speaker_name, text) のリストから esd.list を書き出す。
    複数話者用。フォーマット: path|speaker|language|text（4列）
    """
    esd_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(esd_list_path, "w", encoding="utf-8") as f:
        for path_str, speaker_name, text in entries:
            f.write(f"{path_str}|{speaker_name}|{language}|{text}\n")
    print(f"esd.listを作成しました: {esd_list_path}")


def create_esd_list(
    esd_list_path: Path,
    raw_dir: Path,
    transcript_map: Dict[str, str],
    raw_audio_map: Dict[str, Path],
    speaker_name: str,
    language: str,
) -> None:
    """
    esd.listファイルを作成。
    フォーマット: path|speaker|language|text（4列）

    Args:
        esd_list_path: esd.list の出力パス
        raw_dir: raw ディレクトリのパス
        transcript_map: ファイル名ベース -> 書き起こしテキストのマッピング
        raw_audio_map: raw/ディレクトリ内の音声ファイルのマッピング（ファイル名ベース -> パス）
        speaker_name: 話者名
        language: 言語ID
    """

    missing = [fb for fb in transcript_map if fb not in raw_audio_map]
    if missing:
        _warn_missing_audio(speaker_name, missing, raw_audio_map)

    esd_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(esd_list_path, "w", encoding="utf-8") as f:
        for filename_base, text in sorted(transcript_map.items()):
            if filename_base not in raw_audio_map:
                continue

            audio_file = raw_audio_map[filename_base]

            try:
                relative_path = audio_file.relative_to(raw_dir)
            except ValueError:
                relative_path = Path(audio_file.name)

            relative_path_wav = relative_path.with_suffix(".wav")
            path_str = str(relative_path_wav).replace("\\", "/")

            line = f"{path_str}|{speaker_name}|{language}|{text}\n"
            f.write(line)

    print(f"esd.listを作成しました: {esd_list_path}")


def copy_audio_files(
    audio_dir: Path,
    raw_dir: Path,
    audio_map: Dict[str, Path],
) -> Dict[str, Path]:
    """
    音声ファイルをraw/ディレクトリにコピーする。

    Args:
        audio_dir: 元の音声ファイルディレクトリ
        raw_dir: コピー先のrawディレクトリ
        audio_map: 音声ファイルのマッピング

    Returns:
        raw/ディレクトリ内の音声ファイルのマッピング（ファイル名ベース -> パス）
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    raw_audio_map = {}
    
    for filename_base, audio_file in audio_map.items():
        # raw/ディレクトリに同じファイル名で配置
        dest_path = raw_dir / audio_file.name
        
        if dest_path.exists():
            # print(f"警告: ファイルが既に存在します（スキップ）: {dest_path}")
            raw_audio_map[filename_base] = dest_path
            continue
        
        try:
            shutil.copy2(audio_file, dest_path)
            # print(f"コピーしました: {audio_file.name}")
            raw_audio_map[filename_base] = dest_path
        except Exception as e:
            print(f"エラー: {audio_file.name}のコピーに失敗しました: {e}")
    
    return raw_audio_map


def main():
    parser = argparse.ArgumentParser(
        description='音声データと書き起こしテキストをStyle-Bert-VITS2のフォーマットに変換'
    )
    parser.add_argument(
        '--input_dirpath',
        type=str,
        default=None,
        help='入力用Data相当ディレクトリ（デフォルト: スクリプト所在/Data）'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='モデル名（Data/配下のディレクトリ名）'
    )
    parser.add_argument(
        '--speaker_name',
        type=str,
        default=None,
        help='話者名（デフォルト: モデル名）'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='JP',
        choices=['JP', 'EN', 'ZH'],
        help='言語ID（デフォルト: JP）'
    )
    parser.add_argument(
        '--transcript_file',
        type=str,
        default='transcript_utf8.txt',
        help='書き起こしファイル名（デフォルト: transcript_utf8.txt）'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        default='wav',
        help='音声ファイルが格納されているディレクトリ名（デフォルト: wav）'
    )
    parser.add_argument(
        '--filter-phoneme-match',
        action='store_true',
        help='音素マッチするケースのみ変換する（JP のみ対応）。書き起こしは 2列 または 3列（filename:漢字かな:カタカナ）'
    )

    args = parser.parse_args()

    if args.filter_phoneme_match and args.language != "JP":
        print("エラー: --filter-phoneme-match は言語 JP のみ対応しています。", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(__file__).parent
    input_data_dir = Path(args.input_dirpath) if args.input_dirpath else base_dir / "Data"
    model_dir = input_data_dir / args.model_name

    if not model_dir.exists():
        print(f"エラー: モデルディレクトリが見つかりません: {model_dir}")
        return

    output_raw_dir = base_dir / "Data" / "raw"
    output_esd_list_path = base_dir / "Data" / "esd.list"
    speaker_dirs = get_speaker_subdirs(
        model_dir, args.transcript_file, args.audio_dir
    )

    if speaker_dirs:
        # 複数話者: 話者名ディレクトリごとに wav と transcript を処理
        print(f"複数話者モード: 話者 {speaker_dirs}")
        esd_entries: List[Tuple[str, str, str]] = []
        for speaker_name in speaker_dirs:
            speaker_path = model_dir / speaker_name
            transcript_path = speaker_path / args.transcript_file
            audio_dir_path = speaker_path / args.audio_dir
            print(f"\n話者: {speaker_name}")
            if args.filter_phoneme_match:
                cases = parse_transcript_to_cases(transcript_path)
                from tests.phoneme_matching_filter import filter_cases_by_phoneme_match

                filtered = filter_cases_by_phoneme_match(cases, match=True)
                transcript_map = {f: k for f, k, _ in filtered}
                print(f"  書き起こしエントリ数: {len(cases)} → 音素一致: {len(transcript_map)}")
            else:
                transcript_map = parse_transcript(transcript_path)
                print(f"  書き起こしエントリ数: {len(transcript_map)}")
            audio_map = get_audio_files(audio_dir_path)
            if args.filter_phoneme_match:
                print("audio keys sample:", sorted(audio_map.keys())[:5])
                print("transcript keys sample:", sorted(transcript_map.keys())[:5])
                print("intersection size:", len(set(audio_map.keys()) & set(transcript_map.keys())))
                audio_map = {k: v for k, v in audio_map.items() if k in transcript_map}
            print(f"  音声ファイル数: {len(audio_map)}")
            raw_speaker_dir = output_raw_dir / speaker_name
            print(f"  音声を {raw_speaker_dir} にコピー中...")
            raw_audio_map = copy_audio_files(
                audio_dir_path, raw_speaker_dir, audio_map
            )
            missing = [
                fb for fb, _ in sorted(transcript_map.items())
                if fb not in raw_audio_map
            ]
            if missing:
                _warn_missing_audio(speaker_name, missing, raw_audio_map)
            for filename_base, text in sorted(transcript_map.items()):
                if filename_base not in raw_audio_map:
                    continue
                path_str = f"{speaker_name}/{filename_base}.wav"
                path_str = path_str.replace("\\", "/")
                esd_entries.append((path_str, speaker_name, text))
        print(f"\nesd.listを作成中...")
        write_esd_list(output_esd_list_path, esd_entries, args.language)
    else:
        # 単一話者: モデルディレクトリ直下に transcript と wav/
        speaker_name = args.speaker_name if args.speaker_name else args.model_name
        transcript_path = model_dir / args.transcript_file
        if not transcript_path.exists():
            print(f"エラー: 書き起こしファイルが見つかりません: {transcript_path}")
            return
        print(f"書き起こしファイルを読み込み中: {transcript_path}")
        if args.filter_phoneme_match:
            cases = parse_transcript_to_cases(transcript_path)
            from tests.phoneme_matching_filter import filter_cases_by_phoneme_match

            filtered = filter_cases_by_phoneme_match(cases, match=True)
            transcript_map = {f: k for f, k, _ in filtered}
            print(f"書き起こしエントリ数: {len(cases)} → 音素一致: {len(transcript_map)}")
        else:
            transcript_map = parse_transcript(transcript_path)
            print(f"書き起こしエントリ数: {len(transcript_map)}")
        audio_dir_path = model_dir / args.audio_dir
        if not audio_dir_path.exists():
            print(f"エラー: 音声ディレクトリが見つかりません: {audio_dir_path}")
            return
        print(f"音声ファイルを検索中: {audio_dir_path}")
        audio_map = get_audio_files(audio_dir_path)
        
        if args.filter_phoneme_match:
            print("audio keys sample:", sorted(audio_map.keys())[:5])
            print("transcript keys sample:", sorted(transcript_map.keys())[:5])
            print("intersection size:", len(set(audio_map.keys()) & set(transcript_map.keys())))
            audio_map = {k: v for k, v in audio_map.items() if k in transcript_map}
        print(f"音声ファイル数: {len(audio_map)}")
        raw_speaker_dir = output_raw_dir / speaker_name
        print(f"\n音声ファイルを{raw_speaker_dir}にコピー中...")
        raw_audio_map = copy_audio_files(
            audio_dir_path, raw_speaker_dir, audio_map
        )
        print(f"\nesd.listを作成中...")
        create_esd_list(
            output_esd_list_path,
            output_raw_dir,
            transcript_map,
            raw_audio_map,
            speaker_name,
            args.language,
        )

    print("\n変換が完了しました！")
    print(f"入力モデルディレクトリ: {model_dir}")
    print(f"raw/ディレクトリ: {output_raw_dir}")
    print(f"esd.list: {output_esd_list_path}")


if __name__ == '__main__':
    main()

