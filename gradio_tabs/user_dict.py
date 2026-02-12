"""ユーザー辞書管理の Gradio UI"""

import csv
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gradio as gr
import jaconv
from fastapi import HTTPException

from config import get_path_config
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.nlp.japanese.user_dict import (
    apply_word,
    compiled_dict_path,
    default_dict_path,
    delete_word,
    read_dict,
    rewrite_word,
)
from style_bert_vits2.nlp.japanese.user_dict.part_of_speech_data import (
    part_of_speech_data,
)
from style_bert_vits2.nlp.japanese.user_dict.word_model import WordTypes
from tests.phoneme_matching_filter import (
    format_phoneme_diff,
    is_effective_match,
)
from tests.support.case_loader import load_cases

from convert_data_format import (
    copy_audio_files,
    create_esd_list,
    get_audio_files,
    get_speaker_subdirs,
    parse_transcript_to_cases,
    write_esd_list,
)

# ---------------------------------------------------------------------------
# 軽量な音素列取得（pyopenjtalk のみ、BERT 不使用）
# ---------------------------------------------------------------------------
# g2p.g2p() は以下の理由で重い:
#   1. pyopenjtalk を 2 回呼ぶ（アクセント取得用 + 読み取得用）
#   2. word2ph 計算のために bert_models.load_tokenizer() を呼ぶ
# 音素マッチング検証ではアクセントも word2ph も不要なため、
# pyopenjtalk のラベル出力から音素だけを抽出する軽量版を用意する。
# ---------------------------------------------------------------------------

_P3_PATTERN = re.compile(r"\-(.*?)\+")

# 句読点・感嘆符等。音素比較時に除去する対象。
_PUNCT_SET = frozenset(("!", "?", "…", ",", ".", "'", "-"))

_CARDS_PER_PAGE = 10


def _get_phones_light(text: str) -> list[str]:
    """pyopenjtalk のフロントエンドのみで音素列を取得する軽量版。

    g2p() と同じ音素表記を返すが、アクセント・word2ph 計算を省略し、
    BERT トークナイザのロードも行わない。
    句読点・感嘆符等は音素ではないため除去する。
    """
    norm_text = normalize_text(text)
    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(norm_text))
    phones: list[str] = ["_"]
    for lab in labels:
        match = _P3_PATTERN.search(lab)
        if match is None:
            continue
        p3 = match.group(1)
        if p3 == "sil" or p3 == "pau":
            continue
        if p3 in _PUNCT_SET:
            continue
        # 無声母音 → 有声母音
        if p3 in "AEIOU":
            p3 = p3.lower()
        # 促音
        if p3 == "cl":
            p3 = "q"
        phones.append(p3)
    phones.append("_")
    return phones


# ---------------------------------------------------------------------------
# 並列音素マッチング (ProcessPoolExecutor ワーカー)
# ---------------------------------------------------------------------------
# pyopenjtalk ワーカーサーバーは TCP ソケットでリクエストを逐次処理するため、
# 大量のケースを検証すると IPC がボトルネックになる。
# fork コンテキストで子プロセスを生成し、各プロセスで pyopenjtalk を直接
# インポート・使用することで IPC を排除しつつ複数 CPU コアを活用する。
# ---------------------------------------------------------------------------

_PHONEME_POOL_WORKERS = min(os.cpu_count() or 4, 8)

_worker_ojt = None  # pyopenjtalk module — ワーカープロセスで初期化
_worker_suppress = None  # suppress_pyopenjtalk_stderr — ワーカープロセスで初期化


def _phoneme_worker_init(dict_path_str: str) -> None:
    """ProcessPoolExecutor ワーカーで pyopenjtalk + ユーザー辞書を初期化する。"""
    global _worker_ojt, _worker_suppress
    import pyopenjtalk
    from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_common import (
        suppress_pyopenjtalk_stderr,
    )

    _worker_ojt = pyopenjtalk
    _worker_suppress = suppress_pyopenjtalk_stderr
    if dict_path_str and Path(dict_path_str).is_file():
        with suppress_pyopenjtalk_stderr():
            pyopenjtalk.update_global_jtalk_with_user_dict(dict_path_str)


def _get_phones_in_worker(text: str) -> list[str]:
    """ワーカープロセスで直接 pyopenjtalk を呼び出して音素列を取得する。

    _get_phones_light と同一のロジック。IPC を介さず直接呼び出す。
    """
    norm_text = normalize_text(text)
    with _worker_suppress():
        labels = _worker_ojt.make_label(_worker_ojt.run_frontend(norm_text))
    phones: list[str] = ["_"]
    for lab in labels:
        m = _P3_PATTERN.search(lab)
        if m is None:
            continue
        p3 = m.group(1)
        if p3 == "sil" or p3 == "pau":
            continue
        if p3 in _PUNCT_SET:
            continue
        if p3 in "AEIOU":
            p3 = p3.lower()
        if p3 == "cl":
            p3 = "q"
        phones.append(p3)
    phones.append("_")
    return phones


def _phoneme_check_one(
    case: tuple[str, str],
) -> tuple[list[str], list[str], bool, str | None]:
    """ワーカープロセスで 1 ケースの音素マッチングを実行する。

    Returns:
        (phones_kanji, phones_kata, is_match, error_msg)
        エラー時は phones が空リスト、error_msg にエラー内容を格納。
    """
    kanji_kana, kata = case
    try:
        pk = [p for p in _get_phones_in_worker(kanji_kana) if p not in _PUNCT_SET]
        pkt = [
            p
            for p in _get_phones_in_worker(
                jaconv.kata2hira(kata).replace("は", "ハ")
            )
            if p not in _PUNCT_SET
        ]
        matched = is_effective_match(pk, pkt)
        return pk, pkt, matched, None
    except Exception as e:
        return [], [], False, str(e)


def _create_phoneme_pool() -> ProcessPoolExecutor:
    """音素マッチング用の ProcessPoolExecutor を生成する。"""
    dict_path = str(compiled_dict_path) if compiled_dict_path.is_file() else ""
    ctx = multiprocessing.get_context("fork")
    return ProcessPoolExecutor(
        max_workers=_PHONEME_POOL_WORKERS,
        mp_context=ctx,
        initializer=_phoneme_worker_init,
        initargs=(dict_path,),
    )


_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus")


def _find_audio_path(transcript_path: str, filename: str) -> str | None:
    """transcript_utf8.txt と同階層の wav/ から音声ファイルを探す。

    filename は拡張子付き (e.g. "audio.wav") でも拡張子なし (e.g. "audio")
    でも対応する。Gradio のファイル配信のため絶対パスを返す。
    """
    wav_dir = Path(transcript_path).parent / "wav"
    if not wav_dir.is_dir():
        return None
    stem = Path(filename).stem
    for ext in _AUDIO_EXTENSIONS:
        candidate = wav_dir / f"{stem}{ext}"
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _replace_transcript_entry(
    transcript_path: str,
    filename: str,
    old_kanji_kana: str,
    old_kata: str,
    new_kanji_kana: str,
    new_kata: str,
) -> str:
    """transcript_utf8.txt の指定ケースのテキストやカタカナ読みを置換する。

    意図しない置換を防ぐため、filename:old_kanji_kana:old_kata の行全体で照合する。
    """
    new_kanji_kana = new_kanji_kana.strip()
    new_kata = new_kata.strip()
    if not new_kanji_kana:
        return "エラー: 空のテキストは指定できません。"
    if not new_kata:
        return "エラー: 空のカタカナは指定できません。"
    if old_kanji_kana == new_kanji_kana and old_kata == new_kata:
        return "変更なし: テキスト・カタカナともに同一です。"

    path = Path(transcript_path)
    if not path.is_file():
        return f"エラー: ファイルが見つかりません — {path}"

    raw = path.read_bytes()
    text = raw.decode("utf-8")
    target = f"{filename}:{old_kanji_kana}:{old_kata}"
    replacement = f"{filename}:{new_kanji_kana}:{new_kata}"

    count = text.count(target)
    if count == 0:
        return "エラー: 対象行が見つかりませんでした（既に変更済みの可能性があります）"
    if count > 1:
        return (
            f"エラー: 同一行が {count} 件見つかりました。"
            "安全のため置換を中止します。"
        )

    new_text = text.replace(target, replacement, 1)
    path.write_bytes(new_text.encode("utf-8"))

    changes: list[str] = []
    if old_kanji_kana != new_kanji_kana:
        changes.append(f"テキスト→「{new_kanji_kana}」")
    if old_kata != new_kata:
        changes.append(f"カタカナ→「{new_kata}」")
    return f"更新完了: {filename} の{', '.join(changes)}に変更しました。"


# 品詞の表示ラベルとWordTypesの対応
WORD_TYPE_LABELS: dict[WordTypes, str] = {
    WordTypes.PROPER_NOUN: "固有名詞",
    WordTypes.COMMON_NOUN: "普通名詞",
    WordTypes.VERB: "動詞",
    WordTypes.ADJECTIVE: "形容詞",
    WordTypes.SUFFIX: "接尾辞",
}

LABEL_TO_WORD_TYPE: dict[str, WordTypes] = {v: k for k, v in WORD_TYPE_LABELS.items()}


def _context_id_to_label(context_id: int) -> str:
    """context_id から品詞ラベルを逆引きする"""
    for wt, detail in part_of_speech_data.items():
        if detail.context_id == context_id:
            return WORD_TYPE_LABELS.get(wt, "不明")
    return "不明"


def _load_default_csv_entries() -> list[list[str]]:
    """dict_data/default.csv からエントリを読み込み、表示用リストとして返す"""
    entries: list[list[str]] = []
    if not default_dict_path.is_file():
        return entries
    with open(default_dict_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 15:
                continue
            surface = row[0]
            pronunciation = row[12]
            accent_info = row[13]
            accent_type = accent_info.split("/")[0] if "/" in accent_info else accent_info
            pos = row[4]
            pos_detail = row[5]
            entries.append(
                [
                    "デフォルト",
                    surface,
                    pronunciation,
                    accent_type,
                    f"{pos}/{pos_detail}",
                    "-",
                ]
            )
    return entries


def _load_user_dict_entries() -> list[list[str]]:
    """dict_data/user_dict.json からエントリを読み込み、表示用リストとして返す"""
    entries: list[list[str]] = []
    user_dict = read_dict()
    for _word_uuid, word in user_dict.items():
        word_type_label = _context_id_to_label(word.context_id)
        entries.append(
            [
                "ユーザー",
                word.surface,
                word.pronunciation,
                str(word.accent_type),
                word_type_label,
                str(word.priority),
            ]
        )
    return entries


def _get_all_entries() -> list[list[str]]:
    """デフォルト辞書とユーザー辞書の全エントリを結合して返す"""
    return _load_default_csv_entries() + _load_user_dict_entries()


def _get_user_dict_choices() -> list[tuple[str, str]]:
    """ユーザー辞書エントリのドロップダウン選択肢 (label, uuid) を返す"""
    user_dict = read_dict()
    choices: list[tuple[str, str]] = []
    for word_uuid, word in user_dict.items():
        label = f"{word.surface} → {word.pronunciation}"
        choices.append((label, word_uuid))
    return choices


_ALL_MODELS_LABEL = "(すべて)"


def _get_dataset_root() -> Path:
    """configs/paths.yml の dataset_root を返す"""
    path_config = get_path_config()
    return Path(path_config.dataset_root)


def _get_model_name_choices() -> list[str]:
    """Data ディレクトリ直下のモデル名一覧を取得する"""
    data_dir = _get_dataset_root()
    if not data_dir.is_dir():
        return [_ALL_MODELS_LABEL]
    models = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    return [_ALL_MODELS_LABEL] + models


def _filter_cases_light(
    cases: list[tuple[str, str, str]],
    progress: gr.Progress | None = None,
    label: str = "",
    executor: ProcessPoolExecutor | None = None,
) -> list[tuple[str, str, str]]:
    """音素マッチング検証と同じ軽量フィルタで有効なケースだけを返す。

    _get_phones_light + is_effective_match を使い、verify_phoneme_matching と
    同一のフィルタ条件を適用する。
    executor が渡された場合は並列実行する。
    """
    total = len(cases)
    if total == 0:
        return []

    if executor is not None:
        work_items = [(kk, kt) for _, kk, kt in cases]
        result: list[tuple[str, str, str]] = []
        for i, (check, case) in enumerate(
            zip(executor.map(_phoneme_check_one, work_items, chunksize=32), cases)
        ):
            if progress is not None and ((i + 1) % 50 == 0 or i == total - 1):
                progress(
                    (i + 1) / total,
                    desc=f"{label}: 音素フィルタ中… ({i + 1}/{total})",
                )
            _, _, matched, _ = check
            if matched:
                result.append(case)
        return result

    # 逐次実行（フォールバック）
    result = []
    for i, (filename, kanji_kana, kata) in enumerate(cases):
        if progress is not None and total > 0:
            progress(i / total, desc=f"{label}: 音素フィルタ中… ({i}/{total})")
        try:
            phones_kanji = [
                p for p in _get_phones_light(kanji_kana) if p not in _PUNCT_SET
            ]
            phones_kata = [
                p
                for p in _get_phones_light(
                    jaconv.kata2hira(kata).replace("は", "ハ")
                )
                if p not in _PUNCT_SET
            ]
        except Exception:
            continue
        if is_effective_match(phones_kanji, phones_kata):
            result.append((filename, kanji_kana, kata))
    return result


def create_user_dict_app() -> gr.Blocks:
    word_type_choices = list(WORD_TYPE_LABELS.values())

    # ------------------------------------------------------------------
    # コールバック関数
    # ------------------------------------------------------------------

    def refresh_table():
        return (
            _get_all_entries(),
            gr.update(choices=_get_user_dict_choices(), value=None),
        )

    def add_word(
        surface: str,
        pronunciation: str,
        accent_type: int,
        word_type_label: str,
        priority: int,
    ):
        if not surface or not pronunciation:
            return (
                "表層形と読み（カタカナ）を入力してください。",
                _get_all_entries(),
                gr.update(choices=_get_user_dict_choices(), value=None),
            )
        try:
            word_type = LABEL_TO_WORD_TYPE.get(word_type_label, WordTypes.PROPER_NOUN)
            apply_word(
                surface=surface,
                pronunciation=pronunciation,
                accent_type=int(accent_type),
                word_type=word_type,
                priority=int(priority),
            )
            return (
                f"「{surface}」を追加しました。",
                _get_all_entries(),
                gr.update(choices=_get_user_dict_choices(), value=None),
            )
        except HTTPException as e:
            return (
                f"エラー: {e.detail}",
                _get_all_entries(),
                gr.update(choices=_get_user_dict_choices(), value=None),
            )
        except (ValueError, AssertionError) as e:
            return (
                f"エラー: {e}",
                _get_all_entries(),
                gr.update(choices=_get_user_dict_choices(), value=None),
            )

    def load_for_edit(word_uuid: str | None):
        if not word_uuid:
            return "", "", 0, "固有名詞", 5
        user_dict = read_dict()
        if word_uuid not in user_dict:
            return "", "", 0, "固有名詞", 5
        word = user_dict[word_uuid]
        word_type_label = _context_id_to_label(word.context_id)
        return (
            word.surface,
            word.pronunciation,
            word.accent_type,
            word_type_label,
            word.priority,
        )

    def edit_word(
        word_uuid: str | None,
        surface: str,
        pronunciation: str,
        accent_type: int,
        word_type_label: str,
        priority: int,
    ):
        if not word_uuid:
            return (
                "編集する単語を選択してください。",
                _get_all_entries(),
                gr.update(choices=_get_user_dict_choices(), value=None),
            )
        if not surface or not pronunciation:
            return (
                "表層形と読み（カタカナ）を入力してください。",
                _get_all_entries(),
                gr.update(choices=_get_user_dict_choices(), value=None),
            )
        try:
            word_type = LABEL_TO_WORD_TYPE.get(word_type_label, WordTypes.PROPER_NOUN)
            rewrite_word(
                word_uuid=word_uuid,
                surface=surface,
                pronunciation=pronunciation,
                accent_type=int(accent_type),
                word_type=word_type,
                priority=int(priority),
            )
            return (
                f"「{surface}」を更新しました。",
                _get_all_entries(),
                gr.update(choices=_get_user_dict_choices(), value=None),
            )
        except HTTPException as e:
            return (
                f"エラー: {e.detail}",
                _get_all_entries(),
                gr.update(choices=_get_user_dict_choices(), value=None),
            )
        except (ValueError, AssertionError) as e:
            return (
                f"エラー: {e}",
                _get_all_entries(),
                gr.update(choices=_get_user_dict_choices(), value=None),
            )

    def del_word(word_uuid: str | None):
        if not word_uuid:
            return (
                "削除する単語を選択してください。",
                _get_all_entries(),
                gr.update(choices=_get_user_dict_choices(), value=None),
            )
        try:
            user_dict = read_dict()
            surface = (
                user_dict[word_uuid].surface if word_uuid in user_dict else "(不明)"
            )
            delete_word(word_uuid)
            return (
                f"「{surface}」を削除しました。",
                _get_all_entries(),
                gr.update(choices=_get_user_dict_choices(), value=None),
            )
        except HTTPException as e:
            return (
                f"エラー: {e.detail}",
                _get_all_entries(),
                gr.update(choices=_get_user_dict_choices(), value=None),
            )
        except (ValueError, AssertionError) as e:
            return (
                f"エラー: {e}",
                _get_all_entries(),
                gr.update(choices=_get_user_dict_choices(), value=None),
            )

    # ------------------------------------------------------------------
    # 音素マッチング検証コールバック
    # ------------------------------------------------------------------

    def verify_phoneme_matching(
        model_name: str | None,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[str, dict]:
        data_dir = _get_dataset_root()

        if model_name and model_name != _ALL_MODELS_LABEL:
            target_dir = data_dir / model_name
        else:
            target_dir = data_dir

        if not target_dir.is_dir():
            return f"ディレクトリが見つかりません: {target_dir}", {"entries": [], "page": 0}

        transcript_files = sorted(target_dir.rglob("transcript_utf8.txt"))
        if not transcript_files:
            return (
                f"{target_dir} 以下に transcript_utf8.txt が見つかりませんでした。",
                {"entries": [], "page": 0},
            )

        # 全ファイルからケースを収集
        all_cases: list[tuple[Path, str, str, str]] = []
        for tf in transcript_files:
            for filename, kanji_kana, kata in load_cases(tf):
                all_cases.append((tf, filename, kanji_kana, kata))

        total = len(all_cases)
        if total == 0:
            return "有効なケースが 0 件でした。", {"entries": [], "page": 0}

        # 並列で音素マッチングを実行
        progress(0, desc=f"音素変換中… (0/{total})")
        work_items = [(kk, kt) for _, _, kk, kt in all_cases]

        results: list[tuple[list[str], list[str], bool, str | None]] = []
        with _create_phoneme_pool() as pool:
            for i, result in enumerate(
                pool.map(_phoneme_check_one, work_items, chunksize=32)
            ):
                results.append(result)
                if (i + 1) % 50 == 0 or i == total - 1:
                    progress(
                        (i + 1) / total,
                        desc=f"検証中… ({i + 1}/{total})",
                    )

        # 結果を集計
        valid_count = 0
        mismatch_entries: list[dict] = []
        speaker_stats: dict[str, dict[str, int]] = {}

        for (tf, filename, kanji_kana, kata), (pk, pkt, matched, err) in zip(
            all_cases, results
        ):
            rel_parent = tf.parent.relative_to(target_dir)
            speaker = (
                str(rel_parent) if str(rel_parent) != "." else target_dir.name
            )
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {"valid": 0, "invalid": 0}

            if err is not None:
                speaker_stats[speaker]["invalid"] += 1
                mismatch_entries.append(
                    {
                        "transcript_file": str(tf),
                        "rel_path": str(tf.relative_to(data_dir)),
                        "filename": filename,
                        "kanji_kana": kanji_kana,
                        "kata_original": kata,
                        "audio_path": _find_audio_path(str(tf), filename),
                        "diff_text": "",
                        "mismatch_count": 0,
                        "error": err,
                        "update_status": "",
                    }
                )
            elif matched:
                valid_count += 1
                speaker_stats[speaker]["valid"] += 1
            else:
                speaker_stats[speaker]["invalid"] += 1
                lines, mc = format_phoneme_diff(pk, pkt)
                mismatch_entries.append(
                    {
                        "transcript_file": str(tf),
                        "rel_path": str(tf.relative_to(data_dir)),
                        "filename": filename,
                        "kanji_kana": kanji_kana,
                        "kata_original": kata,
                        "audio_path": _find_audio_path(str(tf), filename),
                        "diff_text": "\n".join(lines),
                        "mismatch_count": mc,
                        "error": None,
                        "update_status": "",
                    }
                )

        invalid_count = len(mismatch_entries)
        progress(1.0, desc="検証完了")

        summary = (
            f"検証完了\n"
            f"  対象ディレクトリ: {target_dir}\n"
            f"  合計: {total} 件\n"
            f"  有効（一致）: {valid_count} 件\n"
            f"  無効（不一致）: {invalid_count} 件\n"
            f"\n"
            f"  話者ごとの内訳:"
        )
        for spk in sorted(speaker_stats):
            stats = speaker_stats[spk]
            summary += (
                f"\n    {spk}: "
                f"有効 {stats['valid']} / 無効 {stats['invalid']}"
                f" (計 {stats['valid'] + stats['invalid']})"
            )
        return summary, {"entries": mismatch_entries, "page": 0}

    # ------------------------------------------------------------------
    # データ変換コールバック（音素フィルタ適用）
    # ------------------------------------------------------------------

    def run_convert_data_format(
        model_name: str | None,
        speaker_name: str,
        language: str,
        input_dirpath: str,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[str, str]:
        """convert_data_format のロジックを音素マッチングフィルタ付きで実行する。

        言語が JP の場合、音素マッチング検証と同じフィルタ
        (_get_phones_light + is_effective_match) を適用し、
        有効と判定されたケースのみを変換対象とする。

        input_dirpath が指定された場合、入力データはそのディレクトリから
        読み込み、出力はデフォルトの dataset_root 配下に書き出す。
        """
        if not model_name:
            return "変換対象のモデル名を選択してください。", ""

        if input_dirpath and input_dirpath.strip():
            input_data_dir = Path(input_dirpath.strip())
            if not input_data_dir.is_dir():
                return (
                    f"入力ディレクトリが見つかりません: {input_data_dir}",
                    "",
                )
        else:
            input_data_dir = _get_dataset_root()

        input_model_dir = input_data_dir / model_name
        if not input_model_dir.is_dir():
            return (
                f"モデルディレクトリが見つかりません: {input_model_dir}",
                "",
            )

        output_model_dir = _get_dataset_root() / model_name
        raw_dir = output_model_dir / "raw"
        transcript_file = "transcript_utf8.txt"
        audio_dir_name = "wav"
        apply_filter = language == "JP"

        speaker_dirs = get_speaker_subdirs(
            input_model_dir, transcript_file, audio_dir_name
        )

        log_lines: list[str] = []
        pool = _create_phoneme_pool() if apply_filter else None

        try:
            if speaker_dirs:
                # ---------- 複数話者モード ----------
                log_lines.append(f"複数話者モード: {speaker_dirs}")
                esd_entries: list[tuple[str, str, str]] = []
                total_original = 0
                total_filtered = 0

                for spk in speaker_dirs:
                    spk_path = input_model_dir / spk
                    t_path = spk_path / transcript_file
                    a_path = spk_path / audio_dir_name

                    cases = parse_transcript_to_cases(t_path)
                    total_original += len(cases)

                    if apply_filter:
                        filtered = _filter_cases_light(
                            cases, progress, f"話者 {spk}",
                            executor=pool,
                        )
                        t_map = {f: k for f, k, _ in filtered}
                    else:
                        t_map = {f: k for f, k, _ in cases}
                    total_filtered += len(t_map)

                    log_lines.append(f"\n話者: {spk}")
                    if apply_filter:
                        log_lines.append(
                            f"  書き起こし: {len(cases)}"
                            f" → 音素一致: {len(t_map)}"
                        )
                    else:
                        log_lines.append(f"  書き起こし: {len(t_map)}")

                    a_map = get_audio_files(a_path)
                    if apply_filter:
                        a_map = {
                            k: v for k, v in a_map.items() if k in t_map
                        }
                    log_lines.append(f"  音声ファイル数: {len(a_map)}")

                    raw_spk_dir = raw_dir / spk
                    raw_a_map = copy_audio_files(
                        a_path, raw_spk_dir, a_map
                    )

                    missing = [
                        fb
                        for fb in sorted(t_map)
                        if fb not in raw_a_map
                    ]
                    if missing:
                        log_lines.append(
                            f"  音声なし: {len(missing)} 件"
                            f" (例: {', '.join(missing[:5])})"
                        )

                    for fb, text in sorted(t_map.items()):
                        if fb not in raw_a_map:
                            continue
                        p = f"{spk}/{fb}.wav".replace("\\", "/")
                        esd_entries.append((p, spk, text))

                write_esd_list(output_model_dir / "esd.list", esd_entries, language)
                progress(1.0, desc="変換完了")

                summary = (
                    f"変換完了\n"
                    f"  モデル: {model_name}\n"
                    f"  合計書き起こし: {total_original} 件\n"
                )
                if apply_filter:
                    summary += f"  音素一致: {total_filtered} 件\n"
                summary += (
                    f"  esd.list エントリ: {len(esd_entries)} 件\n"
                    f"  出力: {output_model_dir / 'esd.list'}"
                )
                if input_data_dir != _get_dataset_root():
                    summary += f"\n  入力元: {input_model_dir}"
            else:
                # ---------- 単一話者モード ----------
                spk = (
                    speaker_name.strip()
                    if speaker_name and speaker_name.strip()
                    else model_name
                )
                t_path = input_model_dir / transcript_file
                if not t_path.exists():
                    return (
                        f"書き起こしファイルが見つかりません: {t_path}",
                        "",
                    )

                a_path = input_model_dir / audio_dir_name
                if not a_path.is_dir():
                    return (
                        f"音声ディレクトリが見つかりません: {a_path}",
                        "",
                    )

                cases = parse_transcript_to_cases(t_path)
                if apply_filter:
                    filtered = _filter_cases_light(
                        cases, progress, spk, executor=pool,
                    )
                    t_map = {f: k for f, k, _ in filtered}
                else:
                    t_map = {f: k for f, k, _ in cases}

                log_lines.append(f"単一話者モード: {spk}")
                if apply_filter:
                    log_lines.append(
                        f"書き起こし: {len(cases)}"
                        f" → 音素一致: {len(t_map)}"
                    )
                else:
                    log_lines.append(f"書き起こし: {len(t_map)}")

                a_map = get_audio_files(a_path)
                if apply_filter:
                    a_map = {
                        k: v for k, v in a_map.items() if k in t_map
                    }
                log_lines.append(f"音声ファイル数: {len(a_map)}")

                raw_a_map = copy_audio_files(a_path, raw_dir, a_map)

                esd_count = sum(
                    1 for fb in t_map if fb in raw_a_map
                )
                create_esd_list(
                    output_model_dir / "esd.list",
                    raw_dir,
                    t_map,
                    raw_a_map,
                    spk,
                    language,
                )
                progress(1.0, desc="変換完了")

                summary = (
                    f"変換完了\n"
                    f"  モデル: {model_name}\n"
                    f"  話者: {spk}\n"
                    f"  合計書き起こし: {len(cases)} 件\n"
                )
                if apply_filter:
                    summary += f"  音素一致: {len(t_map)} 件\n"
                summary += (
                    f"  esd.list エントリ: {esd_count} 件\n"
                    f"  出力: {output_model_dir / 'esd.list'}"
                )
                if input_data_dir != _get_dataset_root():
                    summary += f"\n  入力元: {input_model_dir}"

        except Exception as e:
            return (
                f"エラーが発生しました: {e}",
                "\n".join(log_lines),
            )
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        detail_text = "\n".join(log_lines) if log_lines else ""
        return summary, detail_text

    def _refresh_convert_models(input_dirpath: str):
        """入力ディレクトリに応じてモデル名候補を更新する。"""
        if input_dirpath and input_dirpath.strip():
            d = Path(input_dirpath.strip())
        else:
            d = _get_dataset_root()
        if not d.is_dir():
            return gr.update(choices=[], value=None)
        models = sorted(dd.name for dd in d.iterdir() if dd.is_dir())
        return gr.update(choices=models, value=None)

    # ------------------------------------------------------------------
    # UI レイアウト
    # ------------------------------------------------------------------

    with gr.Blocks() as app:
        gr.Markdown(
            "辞書に単語を登録して、読み上げ時の読み方やアクセントを制御できます。\n"
            "- 「デフォルト」エントリは `dict_data/default.csv` で定義されており、"
            "ここでは編集できません。\n"
            "- 「ユーザー」エントリは追加・編集・削除が可能です。\n"
            "- 読みはカタカナで入力してください。\n"
            "- 追加・編集・削除後は辞書が自動的に再コンパイルされ、"
            "音声合成に反映されます。"
        )

        with gr.Row():
            refresh_btn = gr.Button("一覧を更新")

        dict_table = gr.Dataframe(
            headers=["ソース", "表層形", "読み", "アクセント型", "品詞", "優先度"],
            value=_get_all_entries(),
            interactive=False,
            label="登録済み辞書一覧",
        )

        status_msg = gr.Textbox(label="ステータス", interactive=False)

        # --- 新規追加 ---
        with gr.Accordion("単語を追加", open=True):
            with gr.Row():
                add_surface = gr.Textbox(
                    label="表層形（単語）", placeholder="例: 担々麺"
                )
                add_pronunciation = gr.Textbox(
                    label="読み（カタカナ）", placeholder="例: タンタンメン"
                )
            with gr.Row():
                add_accent_type = gr.Number(
                    label="アクセント型", value=0, precision=0
                )
                add_word_type = gr.Dropdown(
                    label="品詞", choices=word_type_choices, value="固有名詞"
                )
                add_priority = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=5,
                    step=1,
                    label="優先度（0〜10、高いほど優先）",
                )
            add_btn = gr.Button("追加", variant="primary")

        # --- 編集・削除 ---
        with gr.Accordion("ユーザー辞書の編集・削除", open=False):
            user_dict_selector = gr.Dropdown(
                label="編集・削除する単語を選択",
                choices=_get_user_dict_choices(),
                interactive=True,
            )
            with gr.Row():
                edit_surface = gr.Textbox(label="表層形（単語）")
                edit_pronunciation = gr.Textbox(label="読み（カタカナ）")
            with gr.Row():
                edit_accent_type = gr.Number(
                    label="アクセント型", value=0, precision=0
                )
                edit_word_type = gr.Dropdown(
                    label="品詞", choices=word_type_choices, value="固有名詞"
                )
                edit_priority = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=5,
                    step=1,
                    label="優先度（0〜10）",
                )
            with gr.Row():
                edit_btn = gr.Button("更新", variant="primary")
                delete_btn = gr.Button("削除", variant="stop")

        # --- 音素マッチング検証 ---
        with gr.Accordion("音素マッチング検証", open=False):
            gr.Markdown(
                "Data ディレクトリ内の `transcript_utf8.txt` を読み込み、"
                "漢字かな交じり文とカタカナ読みの音素列が一致するかを検証します。\n"
                "辞書登録の効果を確認したり、読みの不整合を発見するのに利用できます。"
            )
            with gr.Row():
                verify_model_name = gr.Dropdown(
                    label="モデル名（空欄で Data 全体を対象）",
                    choices=_get_model_name_choices(),
                    value=_ALL_MODELS_LABEL,
                    interactive=True,
                )
                verify_btn = gr.Button("検証を実行", variant="primary")
            verify_summary = gr.Textbox(
                label="検証結果サマリー",
                interactive=False,
                lines=8,
                max_lines=30,
            )
            mismatch_data = gr.State({"entries": [], "page": 0})

            @gr.render(inputs=mismatch_data)
            def render_mismatch_cards(data):
                entries = data.get("entries", [])
                page = data.get("page", 0)

                if not entries:
                    return

                total = len(entries)
                total_pages = max(
                    1, (total + _CARDS_PER_PAGE - 1) // _CARDS_PER_PAGE
                )
                page = min(page, total_pages - 1)
                start = page * _CARDS_PER_PAGE
                end = min(start + _CARDS_PER_PAGE, total)

                gr.Markdown(
                    f"### 不一致の詳細 — {total} 件"
                    f" （{page + 1}/{total_pages} ページ）"
                )

                if total_pages > 1:
                    with gr.Row():
                        prev_btn = gr.Button(
                            "前のページ",
                            interactive=(page > 0),
                            size="sm",
                            scale=1,
                        )
                        next_btn = gr.Button(
                            "次のページ",
                            interactive=(page < total_pages - 1),
                            size="sm",
                            scale=1,
                        )

                    def go_prev(d):
                        d = dict(d)
                        d["page"] = max(0, d.get("page", 0) - 1)
                        return d

                    def go_next(d):
                        d = dict(d)
                        tp = max(
                            1,
                            (len(d.get("entries", [])) + _CARDS_PER_PAGE - 1)
                            // _CARDS_PER_PAGE,
                        )
                        d["page"] = min(tp - 1, d.get("page", 0) + 1)
                        return d

                    prev_btn.click(
                        fn=go_prev,
                        inputs=[mismatch_data],
                        outputs=[mismatch_data],
                    )
                    next_btn.click(
                        fn=go_next,
                        inputs=[mismatch_data],
                        outputs=[mismatch_data],
                    )

                for idx in range(start, end):
                    entry = entries[idx]
                    with gr.Group():
                        if entry.get("error"):
                            gr.Markdown(
                                f"**#{idx + 1}**"
                                f" `{entry['rel_path']}`"
                                f" — `{entry['filename']}`\n\n"
                                f"音素変換エラー: {entry['error']}"
                            )
                            if entry.get("audio_path"):
                                gr.Audio(
                                    value=entry["audio_path"],
                                    label="音声",
                                    interactive=False,
                                )
                        else:
                            gr.Markdown(
                                f"**#{idx + 1}**"
                                f" `{entry['rel_path']}`"
                                f" — `{entry['filename']}`"
                            )
                            if entry.get("audio_path"):
                                gr.Audio(
                                    value=entry["audio_path"],
                                    label="音声",
                                    interactive=False,
                                )
                            text_input = gr.Textbox(
                                value=entry["kanji_kana"],
                                label="テキスト",
                                interactive=True,
                                lines=1,
                            )
                            with gr.Row():
                                kata_input = gr.Textbox(
                                    value=entry["kata_original"],
                                    label="カタカナ読み",
                                    interactive=True,
                                    lines=1,
                                    scale=5,
                                )
                                update_btn = gr.Button(
                                    "更新",
                                    variant="primary",
                                    scale=1,
                                    min_width=80,
                                )
                            with gr.Accordion("音素差分", open=False):
                                gr.Textbox(
                                    value=(
                                        entry["diff_text"]
                                        + f"\n不一致数: {entry['mismatch_count']}"
                                    ),
                                    show_label=False,
                                    interactive=False,
                                    lines=15,
                                    max_lines=15,
                                )
                            if entry.get("update_status"):
                                gr.Markdown(
                                    f"*{entry['update_status']}*"
                                )

                            def _make_update_fn(i):
                                def _do_update(new_text, new_kata, d):
                                    e = d["entries"][i]
                                    result = _replace_transcript_entry(
                                        e["transcript_file"],
                                        e["filename"],
                                        e["kanji_kana"],
                                        e["kata_original"],
                                        new_text,
                                        new_kata,
                                    )
                                    d = dict(d)
                                    d["entries"] = list(d["entries"])
                                    d["entries"][i] = dict(
                                        d["entries"][i]
                                    )
                                    if result.startswith("更新完了"):
                                        d["entries"][i][
                                            "kanji_kana"
                                        ] = new_text.strip()
                                        d["entries"][i][
                                            "kata_original"
                                        ] = new_kata.strip()
                                    d["entries"][i][
                                        "update_status"
                                    ] = result
                                    return d

                                return _do_update

                            update_btn.click(
                                fn=_make_update_fn(idx),
                                inputs=[
                                    text_input,
                                    kata_input,
                                    mismatch_data,
                                ],
                                outputs=[mismatch_data],
                            )

        # --- データ変換（音素フィルタ適用） ---
        with gr.Accordion("データ変換（音素フィルタ適用）", open=False):
            gr.Markdown(
                "Data ディレクトリ内の書き起こしと音声ファイルを"
                " Style-Bert-VITS2 のフォーマットに変換します。\n"
                "音素マッチング検証と同じフィルタを適用し、"
                "有効（一致）と判定されたケースのみを変換します（JP のみ）。\n"
                "変換後は `raw/` ディレクトリに音声がコピーされ、"
                "`esd.list` が生成されます。"
            )
            convert_input_dir = gr.Textbox(
                label="入力ディレクトリ（空欄でデフォルトの Data ディレクトリを使用）",
                placeholder="例: /path/to/custom/Data",
                value="",
            )
            with gr.Row():
                convert_model_name = gr.Dropdown(
                    label="モデル名",
                    choices=[
                        c
                        for c in _get_model_name_choices()
                        if c != _ALL_MODELS_LABEL
                    ],
                    value=None,
                    interactive=True,
                )
                convert_speaker_name = gr.Textbox(
                    label="話者名（単一話者時のみ、空欄でモデル名を使用）",
                    placeholder="例: speaker1",
                )
                convert_language = gr.Dropdown(
                    label="言語",
                    choices=["JP", "EN", "ZH"],
                    value="JP",
                    interactive=True,
                )
            convert_btn = gr.Button("変換を実行", variant="primary")
            convert_summary = gr.Textbox(
                label="変換結果サマリー",
                interactive=False,
                lines=7,
            )
            convert_detail = gr.Textbox(
                label="詳細ログ",
                interactive=False,
                lines=15,
                max_lines=50,
            )

        # ------------------------------------------------------------------
        # イベントバインド
        # ------------------------------------------------------------------

        refresh_btn.click(
            fn=refresh_table,
            outputs=[dict_table, user_dict_selector],
        )

        add_btn.click(
            fn=add_word,
            inputs=[
                add_surface,
                add_pronunciation,
                add_accent_type,
                add_word_type,
                add_priority,
            ],
            outputs=[status_msg, dict_table, user_dict_selector],
        )

        user_dict_selector.change(
            fn=load_for_edit,
            inputs=[user_dict_selector],
            outputs=[
                edit_surface,
                edit_pronunciation,
                edit_accent_type,
                edit_word_type,
                edit_priority,
            ],
        )

        edit_btn.click(
            fn=edit_word,
            inputs=[
                user_dict_selector,
                edit_surface,
                edit_pronunciation,
                edit_accent_type,
                edit_word_type,
                edit_priority,
            ],
            outputs=[status_msg, dict_table, user_dict_selector],
        )

        delete_btn.click(
            fn=del_word,
            inputs=[user_dict_selector],
            outputs=[status_msg, dict_table, user_dict_selector],
        )

        verify_btn.click(
            fn=verify_phoneme_matching,
            inputs=[verify_model_name],
            outputs=[verify_summary, mismatch_data],
        )

        convert_input_dir.change(
            fn=_refresh_convert_models,
            inputs=[convert_input_dir],
            outputs=[convert_model_name],
        )

        convert_btn.click(
            fn=run_convert_data_format,
            inputs=[
                convert_model_name,
                convert_speaker_name,
                convert_language,
                convert_input_dir,
            ],
            outputs=[convert_summary, convert_detail],
        )

    return app
