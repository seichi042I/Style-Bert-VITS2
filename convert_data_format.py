#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音声データと書き起こしテキストをStyle-Bert-VITS2のフォーマットに変換するスクリプト

使用方法:
    python convert_data_format.py --model_name kugimiya_rie --speaker_name kugimiya --language JP

引数:
    --model_name: モデル名（Data/配下のディレクトリ名）
    --speaker_name: 話者名（デフォルト: モデル名）
    --language: 言語ID（JP/EN/ZH、デフォルト: JP）
    --copy: 音声ファイルをコピーする（デフォルト: 移動）
"""

import shutil
import argparse
from pathlib import Path
from typing import Dict


def parse_transcript(transcript_path: Path) -> Dict[str, str]:
    """
    書き起こしファイルを解析して、ファイル名とテキストのマッピングを返す
    
    Args:
        transcript_path: 書き起こしファイルのパス
        
    Returns:
        ファイル名（拡張子なし）をキー、書き起こしテキストを値とする辞書
    """
    transcript_map = {}
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # フォーマット: ファイル名:書き起こしテキスト
            if ':' in line:
                filename, text = line.split(':', 1)
                # 拡張子を除去
                filename_base = Path(filename).stem
                transcript_map[filename_base] = text
    
    return transcript_map


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
        if audio_file.is_file() and audio_file.suffix.lower() in audio_extensions:
            filename_base = audio_file.stem
            audio_map[filename_base] = audio_file
    
    return audio_map


def create_esd_list(
    model_dir: Path,
    transcript_map: Dict[str, str],
    raw_audio_map: Dict[str, Path],
    speaker_name: str,
    language: str
) -> None:
    """
    esd.listファイルを作成
    
    Args:
        model_dir: モデルディレクトリのパス
        transcript_map: 書き起こしテキストのマッピング
        raw_audio_map: raw/ディレクトリ内の音声ファイルのマッピング（ファイル名ベース -> パス）
        speaker_name: 話者名
        language: 言語ID
    """
    esd_list_path = model_dir / 'esd.list'
    raw_dir = model_dir / 'raw'
    
    with open(esd_list_path, 'w', encoding='utf-8') as f:
        # 書き起こしがあるファイルのみを処理
        for filename_base, text in sorted(transcript_map.items()):
            if filename_base not in raw_audio_map:
                print(f"警告: raw/内に音声ファイルが見つかりません: {filename_base}")
                continue
            
            audio_file = raw_audio_map[filename_base]
            
            # raw/からの相対パスを取得
            try:
                relative_path = audio_file.relative_to(raw_dir)
            except ValueError:
                # raw/配下にない場合は、ファイル名のみ
                relative_path = Path(audio_file.name)
            
            # 拡張子を.wavに変更（実際の拡張子に関わらず）
            relative_path_wav = relative_path.with_suffix('.wav')
            
            # パスをスラッシュ区切りに変換（Windowsでも）
            path_str = str(relative_path_wav).replace('\\', '/')
            
            # esd.listのフォーマット: path|speaker|language|text
            line = f"{path_str}|{speaker_name}|{language}|{text}\n"
            f.write(line)
    
    print(f"esd.listを作成しました: {esd_list_path}")


def move_or_copy_audio_files(
    audio_dir: Path,
    raw_dir: Path,
    audio_map: Dict[str, Path],
    copy_mode: bool = False
) -> Dict[str, Path]:
    """
    音声ファイルをraw/ディレクトリに移動またはコピー
    
    Args:
        audio_dir: 元の音声ファイルディレクトリ
        raw_dir: 移動先のrawディレクトリ
        audio_map: 音声ファイルのマッピング
        copy_mode: Trueの場合はコピー、Falseの場合は移動
        
    Returns:
        raw/ディレクトリ内の音声ファイルのマッピング（ファイル名ベース -> パス）
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    operation = "コピー" if copy_mode else "移動"
    func = shutil.copy2 if copy_mode else shutil.move
    
    raw_audio_map = {}
    
    for filename_base, audio_file in audio_map.items():
        # raw/ディレクトリに同じファイル名で配置
        dest_path = raw_dir / audio_file.name
        
        if dest_path.exists():
            print(f"警告: ファイルが既に存在します（スキップ）: {dest_path}")
            raw_audio_map[filename_base] = dest_path
            continue
        
        try:
            func(audio_file, dest_path)
            print(f"{operation}しました: {audio_file.name}")
            raw_audio_map[filename_base] = dest_path
        except Exception as e:
            print(f"エラー: {audio_file.name}の{operation}に失敗しました: {e}")
    
    return raw_audio_map


def main():
    parser = argparse.ArgumentParser(
        description='音声データと書き起こしテキストをStyle-Bert-VITS2のフォーマットに変換'
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
        '--copy',
        action='store_true',
        help='音声ファイルをコピーする（デフォルト: 移動）'
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
    
    args = parser.parse_args()
    
    # パスを設定
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'Data'
    model_dir = data_dir / args.model_name
    
    if not model_dir.exists():
        print(f"エラー: モデルディレクトリが見つかりません: {model_dir}")
        return
    
    # 話者名が指定されていない場合はモデル名を使用
    speaker_name = args.speaker_name if args.speaker_name else args.model_name
    
    # 書き起こしファイルを読み込み
    transcript_path = model_dir / args.transcript_file
    if not transcript_path.exists():
        print(f"エラー: 書き起こしファイルが見つかりません: {transcript_path}")
        return
    
    print(f"書き起こしファイルを読み込み中: {transcript_path}")
    transcript_map = parse_transcript(transcript_path)
    print(f"書き起こしエントリ数: {len(transcript_map)}")
    
    # 音声ファイルを取得
    audio_dir = model_dir / args.audio_dir
    if not audio_dir.exists():
        print(f"エラー: 音声ファイルディレクトリが見つかりません: {audio_dir}")
        return
    
    print(f"音声ファイルを検索中: {audio_dir}")
    audio_map = get_audio_files(audio_dir)
    print(f"音声ファイル数: {len(audio_map)}")
    
    # raw/ディレクトリを作成して音声ファイルを移動/コピー
    raw_dir = model_dir / 'raw'
    print(f"\n音声ファイルを{raw_dir}に{'コピー' if args.copy else '移動'}中...")
    raw_audio_map = move_or_copy_audio_files(audio_dir, raw_dir, audio_map, copy_mode=args.copy)
    
    # esd.listを作成
    print(f"\nesd.listを作成中...")
    create_esd_list(model_dir, transcript_map, raw_audio_map, speaker_name, args.language)
    
    print("\n変換が完了しました！")
    print(f"モデルディレクトリ: {model_dir}")
    print(f"raw/ディレクトリ: {raw_dir}")
    print(f"esd.list: {model_dir / 'esd.list'}")


if __name__ == '__main__':
    main()

