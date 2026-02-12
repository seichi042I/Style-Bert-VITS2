import argparse
import atexit

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from config import get_config
from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.nlp import cleaned_text_to_sequence, extract_bert_feature
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_common import WORKER_PORT
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


config = get_config()


# ── ワーカープロセスごとの状態 ─────────────────────────────────────────────────
_worker_device: str = "cpu"


def _init_worker(
    counter: mp.Value,
    base_port: int,
    num_pyopenjtalk_workers: int,
    lang_names: list[str],
    device: str,
    use_multi_device: bool,
) -> None:
    """mp.Pool の各ワーカープロセスで一度だけ実行される初期化関数。

    - 担当の pyopenjtalk サーバーに接続（サーバーはメインプロセスが起動済み）
    - BERT モデル・トークナイザーをプロセス固有にロード
    """
    global _worker_device

    # ワーカー ID 割当
    with counter.get_lock():
        wid = counter.value
        counter.value += 1

    # pyopenjtalk サーバーに接続
    port = base_port + (wid % num_pyopenjtalk_workers)
    pyopenjtalk_worker.initialize_worker(port=port, num_workers=1)

    # ワーカープロセスからはサーバーを終了させない（メインプロセスが管理する）
    atexit.unregister(pyopenjtalk_worker.terminate_worker)

    # デバイス決定
    if use_multi_device and torch.cuda.is_available():
        gpu_id = wid % torch.cuda.device_count()
        _worker_device = f"cuda:{gpu_id}"
    elif device != "cpu" and torch.cuda.is_available():
        _worker_device = device
    else:
        _worker_device = "cpu"

    # BERT モデル・トークナイザーのロード（プロセスごとに独立したコピー）
    from style_bert_vits2.nlp import bert_models

    for name in lang_names:
        lang = Languages[name]
        bert_models.load_model(lang)
        bert_models.load_tokenizer(lang)
        if _worker_device != "cpu":
            bert_models.transfer_model(lang, _worker_device)

    logger.info(f"Worker {wid} ready (port={port}, device={_worker_device})")


def process_line(x: tuple[str, bool]) -> None:
    line, add_blank = x
    device = _worker_device

    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(
        phone, tone, Languages[language_str]
    )

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")

    try:
        bert = torch.load(bert_path, weights_only=True)
        assert bert.shape[-1] == len(phone)
    except Exception:
        try:
            bert = extract_bert_feature(text, word2ph, Languages(language_str), device)
        except ImportError as e:
            if "PyTorch" in str(e) or "torch" in str(e).lower():
                raise ImportError(
                    f"BERT feature extraction failed: {e}. "
                    "With PyTorch < 2.4, use transformers>=4.30,<4.46 (see requirements.txt)."
                ) from e
            raise
        assert bert.shape[-1] == len(phone)
        torch.save(bert, bert_path)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.bert_gen_config.config_path
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel processes for BERT feature generation.",
    )
    args, _ = parser.parse_known_args()
    config_path = args.config
    num_processes: int = args.num_processes

    # pyopenjtalk ワーカーサーバー起動 + 辞書適用
    pyopenjtalk_worker.initialize_worker(num_workers=num_processes)
    update_dict()

    hps = HyperParameters.load_from_json(config_path)
    lines: list[str] = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    add_blank = [hps.data.add_blank] * len(lines)

    # データに含まれる言語を特定
    languages_in_data: list[str] = []
    seen: set[str] = set()
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) >= 4 and parts[2] not in seen:
            try:
                Languages[parts[2]]
                seen.add(parts[2])
                languages_in_data.append(parts[2])
            except KeyError:
                pass

    device = config.bert_gen_config.device
    use_multi_device = config.bert_gen_config.use_multi_device

    if len(lines) != 0:
        worker_counter: mp.Value = mp.Value("i", 0)

        with mp.Pool(
            num_processes,
            initializer=_init_worker,
            initargs=(
                worker_counter,
                WORKER_PORT,
                num_processes,
                languages_in_data,
                device,
                use_multi_device,
            ),
        ) as pool:
            _ = list(
                tqdm(
                    pool.imap_unordered(process_line, zip(lines, add_blank)),
                    total=len(lines),
                    file=SAFE_STDOUT,
                    dynamic_ncols=True,
                )
            )

    logger.info(f"bert.pt is generated! total: {len(lines)} bert.pt files.")
