import argparse
from concurrent.futures import ThreadPoolExecutor

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
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


config = get_config()
# ワーカーの初期化と辞書適用は if __name__ ブロック内で行う
# (num_processes に合わせて並列ワーカーを起動するため)


def process_line(x: tuple[str, bool]):
    line, add_blank = x
    device = config.bert_gen_config.device
    if config.bert_gen_config.use_multi_device:
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = f"cuda:{gpu_id}"
        else:
            device = "cpu"
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
        bert = torch.load(bert_path)
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


preprocess_text_config = config.preprocess_text_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.bert_gen_config.config_path
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel threads for BERT feature generation.",
    )
    args, _ = parser.parse_known_args()
    config_path = args.config
    num_processes: int = args.num_processes

    # num_processes 分のワーカーサーバーを起動し、辞書を全ワーカーに適用
    pyopenjtalk_worker.initialize_worker(num_workers=num_processes)
    update_dict()

    hps = HyperParameters.load_from_json(config_path)
    lines: list[str] = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    add_blank = [hps.data.add_blank] * len(lines)

    if len(lines) != 0:
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            _ = list(
                tqdm(
                    executor.map(process_line, zip(lines, add_blank)),
                    total=len(lines),
                    file=SAFE_STDOUT,
                    dynamic_ncols=True,
                )
            )

    logger.info(f"bert.pt is generated! total: {len(lines)} bert.pt files.")
