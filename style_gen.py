import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# pyannote.audio が use_auth_token を渡すが、huggingface_hub は token に統一済み。
# 互換のため、pyannote 読み込み前に hf_hub_download をパッチする。
import huggingface_hub.file_download as _hf_file_download

_orig_hf_hub_download = _hf_file_download.hf_hub_download


def _patched_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        if "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        else:
            kwargs.pop("use_auth_token")
    return _orig_hf_hub_download(*args, **kwargs)


_hf_file_download.hf_hub_download = _patched_hf_hub_download
# トップレベルから import している場合に備えて同じ参照を差し替える
import huggingface_hub as _hf_hub

if hasattr(_hf_hub, "hf_hub_download"):
    _hf_hub.hf_hub_download = _patched_hf_hub_download

import numpy as np
import torch

# PyTorch 2.6+ で torch.load のデフォルトが weights_only=True に変更された。
# pyannote-audio 3.x のチェックポイントには TorchVersion, Specifications 等
# weights_only=True では許可されないオブジェクトが多数含まれる。
# lightning_fabric が torch.serialization.load を直接参照する場合があるため両方パッチする。
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs = dict(kwargs)
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
torch.serialization.load = _patched_torch_load

from numpy.typing import NDArray
from pyannote.audio import Inference, Model
from tqdm import tqdm

from config import get_config
from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


config = get_config()

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")
device = torch.device(config.style_gen_config.device)
inference.to(device)


class NaNValueError(ValueError):
    """カスタム例外クラス。NaN値が見つかった場合に使用されます。"""


# 推論時にインポートするために短いが関数を書く
def get_style_vector(wav_path: str) -> NDArray[Any]:
    return inference(wav_path)  # type: ignore


def save_style_vector(wav_path: str):
    try:
        style_vec = get_style_vector(wav_path)
    except Exception as e:
        print("\n")
        logger.error(f"Error occurred with file: {wav_path}, Details:\n{e}\n")
        raise
    # 値にNaNが含まれていると悪影響なのでチェックする
    if np.isnan(style_vec).any():
        print("\n")
        logger.warning(f"NaN value found in style vector: {wav_path}")
        raise NaNValueError(f"NaN value found in style vector: {wav_path}")
    np.save(f"{wav_path}.npy", style_vec)  # `test.wav` -> `test.wav.npy`


def _remove_stale_npy(wav_path: str) -> None:
    """以前の実行で残った .npy ファイルがあれば削除する。"""
    npy_path = f"{wav_path}.npy"
    if os.path.exists(npy_path):
        try:
            os.remove(npy_path)
        except OSError as e:
            logger.warning(f"Failed to remove stale npy file {npy_path}: {e}")


def process_line(line: str):
    wav_path = line.split("|")[0]
    try:
        save_style_vector(wav_path)
        return line, None
    except NaNValueError:
        _remove_stale_npy(wav_path)
        return line, "nan_error"
    except Exception as e:
        logger.error(f"Failed to generate style vector for {wav_path}: {e}")
        _remove_stale_npy(wav_path)
        return line, "error"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.style_gen_config.config_path
    )
    parser.add_argument(
        "--num_processes", type=int, default=config.style_gen_config.num_processes
    )
    args, _ = parser.parse_known_args()
    config_path: str = args.config
    num_processes: int = args.num_processes

    hps = HyperParameters.load_from_json(config_path)

    device = config.style_gen_config.device

    def _process_split(
        list_path: str, split_name: str
    ) -> tuple[list[str], list[str]]:
        """指定リストを処理し、(ok_lines, failed_wav_paths) を返す。"""
        lines: list[str] = []
        with open(list_path, encoding="utf-8") as f:
            lines.extend(f.readlines())

        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            results = list(
                tqdm(
                    executor.map(process_line, lines),
                    total=len(lines),
                    file=SAFE_STDOUT,
                    dynamic_ncols=True,
                )
            )

        ok_lines = [line for line, error in results if error is None]
        nan_lines = [line for line, error in results if error == "nan_error"]
        err_lines = [line for line, error in results if error == "error"]
        failed_paths: list[str] = []

        if nan_lines:
            paths = [l.split("|")[0] for l in nan_lines]
            failed_paths.extend(paths)
            logger.warning(
                f"[{split_name}] NaN style vectors in {len(nan_lines)} files — "
                f"excluded from {split_name} data: {paths}"
            )
        if err_lines:
            paths = [l.split("|")[0] for l in err_lines]
            failed_paths.extend(paths)
            logger.warning(
                f"[{split_name}] Failed to generate style vectors for {len(err_lines)} files — "
                f"excluded from {split_name} data: {paths}"
            )

        return ok_lines, failed_paths

    # ── training / validation それぞれ処理 ────────────────────────────
    ok_training_lines, failed_training = _process_split(
        hps.data.training_files, "train"
    )
    ok_val_lines, failed_val = _process_split(
        hps.data.validation_files, "val"
    )

    # ── train.list / val.list を正常行のみで上書き ────────────────────
    with open(hps.data.training_files, "w", encoding="utf-8") as f:
        f.writelines(ok_training_lines)
    with open(hps.data.validation_files, "w", encoding="utf-8") as f:
        f.writelines(ok_val_lines)

    # ── 失敗した wav に対応する .bert.pt も削除（次ステップで再生成されないよう）
    all_failed = set(failed_training + failed_val)
    for wav_path in all_failed:
        bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")
        if os.path.exists(bert_path):
            try:
                os.remove(bert_path)
            except OSError as e:
                logger.warning(f"Failed to remove stale bert file {bert_path}: {e}")

    ok_num = len(ok_training_lines) + len(ok_val_lines)
    fail_num = len(all_failed)
    logger.info(
        f"Finished generating style vectors! "
        f"ok: {ok_num}, excluded: {fail_num}"
    )
