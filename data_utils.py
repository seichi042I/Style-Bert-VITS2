import os
import random
import sys
import threading
from collections import defaultdict
from queue import Queue

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from config import get_config
from mel_processing import mel_spectrogram_torch, spectrogram_torch
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons
from style_bert_vits2.models.hyper_parameters import HyperParametersData
from style_bert_vits2.models.utils import load_filepaths_and_text, load_wav_to_torch
from style_bert_vits2.nlp import cleaned_text_to_sequence


config = get_config()
"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(
        self,
        audiopaths_sid_text: str,
        hparams: HyperParametersData,
        cache_in_memory: bool = False,
    ):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams
        self.use_jp_extra = getattr(hparams, "use_jp_extra", False)

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 384)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

        # In-memory cache: preload all samples to avoid disk I/O during training
        self._cache = None
        if cache_in_memory:
            self._preload_to_memory()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        skipped_reasons: dict[str, list[str]] = {
            "missing_npy": [],
            "nan_style_vec": [],
            "missing_wav": [],
        }
        logger.info("Init dataset...")
        for _id, spk, language, text, phones, tone, word2ph in tqdm(
            self.audiopaths_sid_text, file=sys.stdout, dynamic_ncols=True
        ):
            audiopath = f"{_id}"

            # wav ファイルの存在チェック
            if not os.path.exists(audiopath):
                skipped += 1
                skipped_reasons["missing_wav"].append(audiopath)
                continue

            # style vector (.npy) の存在チェック
            npy_path = f"{audiopath}.npy"
            if not os.path.exists(npy_path):
                skipped += 1
                skipped_reasons["missing_npy"].append(audiopath)
                continue

            # style vector の NaN チェック
            try:
                style_vec = np.load(npy_path)
                if np.isnan(style_vec).any():
                    skipped += 1
                    skipped_reasons["nan_style_vec"].append(audiopath)
                    continue
            except Exception as e:
                logger.warning(f"Failed to load {npy_path}: {e}")
                skipped += 1
                skipped_reasons["missing_npy"].append(audiopath)
                continue

            phones = phones.split(" ")
            tone = [int(i) for i in tone.split(" ")]
            word2ph = [int(i) for i in word2ph.split(" ")]
            audiopaths_sid_text_new.append(
                [audiopath, spk, language, text, phones, tone, word2ph]
            )
            lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))

        for reason, paths in skipped_reasons.items():
            if paths:
                logger.warning(f"Skipped {len(paths)} entries ({reason}): {paths[:10]}{'...' if len(paths) > 10 else ''}")
        logger.info(
            f"skipped: {skipped}, total: {len(self.audiopaths_sid_text)}"
        )
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        # 話者名をインデックスごとに保持（バケットサンプラーの話者均等化に利用）
        self.speakers = [x[1] for x in self.audiopaths_sid_text]

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text

        bert, ja_bert, en_bert, phones, tone, language = self.get_text(
            text, word2ph, phones, tone, language, audiopath
        )

        spec, wav = self.get_audio(audiopath)
        sid = torch.LongTensor([int(self.spk_map[sid])])
        style_vec = torch.FloatTensor(np.load(f"{audiopath}.npy"))
        if self.use_jp_extra:
            return (phones, spec, wav, sid, tone, language, ja_bert, style_vec)
        else:
            return (
                phones,
                spec,
                wav,
                sid,
                tone,
                language,
                bert,
                ja_bert,
                en_bert,
                style_vec,
            )

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{filename} {sampling_rate} SR doesn't match target {self.sampling_rate} SR"
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            spec = torch.load(spec_filename)
        except:
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            if config.train_ms_config.spec_cache:
                torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text, word2ph, phone, tone, language_str, wav_path):
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
        if self.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        bert_path = wav_path.replace(".wav", ".bert.pt")
        try:
            bert_ori = torch.load(bert_path)
            assert bert_ori.shape[-1] == len(phone)
        except Exception as e:
            logger.warning("Bert load Failed")
            logger.warning(e)

        if language_str == "ZH":
            bert = bert_ori
            ja_bert = torch.zeros(1024, len(phone))
            en_bert = torch.zeros(1024, len(phone))
        elif language_str == "JP":
            bert = torch.zeros(1024, len(phone))
            ja_bert = bert_ori
            en_bert = torch.zeros(1024, len(phone))
        elif language_str == "EN":
            bert = torch.zeros(1024, len(phone))
            ja_bert = torch.zeros(1024, len(phone))
            en_bert = bert_ori
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return bert, ja_bert, en_bert, phone, tone, language

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def _preload_to_memory(self):
        """Preload all samples into RAM to eliminate disk I/O during training."""
        n = len(self.audiopaths_sid_text)
        logger.info(f"Preloading {n} samples into RAM...")
        self._cache = [None] * n
        for i in tqdm(
            range(n), desc="Caching to RAM", file=sys.stdout, dynamic_ncols=True
        ):
            self._cache[i] = self.get_audio_text_speaker_pair(
                self.audiopaths_sid_text[i]
            )
        # Estimate memory usage from first sample
        sample = self._cache[0]
        sample_bytes = sum(
            t.nelement() * t.element_size() if isinstance(t, torch.Tensor) else 0
            for t in sample
        )
        est_total_mb = sample_bytes * n / (1024 * 1024)
        logger.info(
            f"Preloaded {n} samples into RAM (estimated ~{est_total_mb:.0f} MB)"
        )

    def __getitem__(self, index):
        if self._cache is not None:
            return self._cache[index]
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


def _round_up(x: int, granularity: int) -> int:
    """Round *x* up to the nearest multiple of *granularity*."""
    return (x + granularity - 1) // granularity * granularity


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False, use_jp_extra=False):
        self.return_ids = return_ids
        self.use_jp_extra = use_jp_extra

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        # Round up to fixed granularity so that tensor shapes are reused
        # across batches.  This stabilizes VRAM usage in three ways:
        #   1. The CUDA caching allocator can reuse memory blocks more often
        #      (same-sized allocations hit the free-list instead of splitting).
        #   2. CuDNN benchmark mode caches fewer algorithm configurations
        #      (fewer unique shapes → fewer workspace allocations).
        #   3. Intermediate activations keep a consistent footprint.
        _PAD_TEXT = 32
        _PAD_SPEC = 32
        _PAD_WAV = 512

        max_text_len = _round_up(max(len(x[0]) for x in batch), _PAD_TEXT)
        max_spec_len = _round_up(max(x[1].size(1) for x in batch), _PAD_SPEC)
        max_wav_len = _round_up(max(x[2].size(1) for x in batch), _PAD_WAV)

        # Lengths / IDs — fully assigned in the loop, no need to zero
        text_lengths = torch.empty(len(batch), dtype=torch.long)
        spec_lengths = torch.empty(len(batch), dtype=torch.long)
        wav_lengths = torch.empty(len(batch), dtype=torch.long)
        sid = torch.empty(len(batch), dtype=torch.long)

        # Padded sequences — zero-initialized
        # NOTE: pin_memory is intentionally NOT used here.
        # For DataLoader path, DataLoader(pin_memory=True) handles pinning
        # via its internal thread (only prefetch_factor batches pinned at a time).
        # For PreCollatedBatchStore, pageable memory is correct — bulk-pinning
        # all batches wastes physical RAM (page-locked memory cannot be swapped).
        text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        tone_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        language_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        bert_padded = torch.zeros(len(batch), 1024, max_text_len)
        if not self.use_jp_extra:
            ja_bert_padded = torch.zeros(len(batch), 1024, max_text_len)
            en_bert_padded = torch.zeros(len(batch), 1024, max_text_len)
        style_vec = torch.zeros(len(batch), 256)

        spec_padded = torch.zeros(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.zeros(len(batch), 1, max_wav_len)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            tone = row[4]
            tone_padded[i, : tone.size(0)] = tone

            language = row[5]
            language_padded[i, : language.size(0)] = language

            bert = row[6]
            bert_padded[i, :, : bert.size(1)] = bert

            if self.use_jp_extra:
                style_vec[i, :] = row[7]
            else:
                ja_bert = row[7]
                ja_bert_padded[i, :, : ja_bert.size(1)] = ja_bert

                en_bert = row[8]
                en_bert_padded[i, :, : en_bert.size(1)] = en_bert
                style_vec[i, :] = row[9]

        if self.use_jp_extra:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                tone_padded,
                language_padded,
                bert_padded,
                style_vec,
            )
        else:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                tone_padded,
                language_padded,
                bert_padded,
                ja_bert_padded,
                en_bert_padded,
                style_vec,
            )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
        max_batch_frames=None,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        # 話者均等化: データセットが話者情報を持つ場合のみ使用
        self.speakers = getattr(dataset, "speakers", None)

        # Per-bucket VRAM budget: バケットごとにバッチサイズを調整して
        # 長いシーケンスでの OOM を防ぐ。アテンション層のメモリ使用量は
        # O(B * L²) で増加するため、B * L を一定に保つことで VRAM 消費を
        # バケット間でおおむね均一にする。
        if max_batch_frames is None:
            # 自動計算: 設定された batch_size は最短バケットに適用し、
            # 長いバケットは比例的に縮小する
            ref_length = boundaries[1] if len(boundaries) > 1 else boundaries[-1]
            self._max_batch_frames = batch_size * ref_length
        elif max_batch_frames <= 0:
            # 明示的に無効化（全バケットで均一の batch_size を使用）
            self._max_batch_frames = None
        else:
            self._max_batch_frames = max_batch_frames

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        logger.info(f"Bucket info: {self.num_samples_per_bucket}")
        bucket_ranges = [
            f"{self.boundaries[i]}-{self.boundaries[i+1]}"
            for i in range(len(self._batch_sizes))
        ]
        logger.info(
            "Per-bucket batch sizes"
            + (f" (max_batch_frames={self._max_batch_frames})"
               if self._max_batch_frames is not None else " (uniform)")
            + ": "
            + ", ".join(
                f"[{r}]={bs}"
                for r, bs in zip(bucket_ranges, self._batch_sizes)
            )
        )

        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
            logger.info("Bucket warning ", e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        # バケットごとのバッチサイズを算出
        self._batch_sizes = []
        for i in range(len(buckets)):
            bucket_max_len = self.boundaries[i + 1]
            if self._max_batch_frames is not None:
                effective_bs = max(
                    1,
                    min(self.batch_size, self._max_batch_frames // bucket_max_len),
                )
            else:
                effective_bs = self.batch_size
            self._batch_sizes.append(effective_bs)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self._batch_sizes[i]
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def _round_robin_order(self, bucket: list) -> list:
        """
        バケット内のインデックスを話者でラウンドロビン（A,B,C,A,B,C,...）に並べ替える。
        同一長さ帯からバッチを作る際に話者が均等に含まれるようにする。
        """
        if not bucket or self.speakers is None:
            return list(range(len(bucket)))
        # 話者ごとにバケット内の位置（bucket の何番目か）をグループ化
        groups = defaultdict(list)
        for pos in range(len(bucket)):
            spk = self.speakers[bucket[pos]]
            groups[spk].append(pos)
        sorted_speakers = sorted(groups.keys())
        # ラウンドロビンで並べた位置のリストを構築
        result = []
        max_len = max(len(groups[spk]) for spk in sorted_speakers)
        for round_idx in range(max_len):
            for spk in sorted_speakers:
                if round_idx < len(groups[spk]):
                    result.append(groups[spk][round_idx])
        return result

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.speakers is not None:
            # 話者をラウンドロビンで均等化した順序を使用
            for bucket in self.buckets:
                indices.append(self._round_robin_order(bucket))
        elif self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]
            batch_size_i = self._batch_sizes[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching (per-bucket batch size for VRAM safety)
            for j in range(len(ids_bucket) // batch_size_i):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * batch_size_i : (j + 1) * batch_size_i
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return sum(
            ns // self.num_replicas // bs
            for ns, bs in zip(self.num_samples_per_bucket, self._batch_sizes)
        )


_PREFETCH_SENTINEL = object()


class _VRAMPrefetchIter:
    """Background-thread iterator: pin (if needed) + async DMA to VRAM.

    A dedicated background thread takes CPU batches from *source_iter*,
    pins any pageable tensors (skips already-pinned ones), and launches
    an asynchronous H2D copy on a **non-default CUDA stream**.  The result
    is queued for the training loop.

    This achieves true copy-compute overlap because:
      1. Source is pinned  →  cudaMemcpyAsync returns immediately.
      2. Transfer is on a separate stream  →  DMA engine and CUDA cores
         work in parallel.
      3. Modern GPUs (Volta+) have ≥ 2 DMA engines.

    The queue depth (*max_prefetch*) controls how many batches reside on
    VRAM ahead of consumption.  2 is enough for double-buffering; more
    just wastes VRAM without improving throughput.
    """

    def __init__(self, source_iter, device, max_prefetch=2):
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._queue: Queue = Queue(maxsize=max_prefetch)
        self._thread = threading.Thread(
            target=self._worker, args=(source_iter,), daemon=True
        )
        self._thread.start()

    # --- producer (background thread) -----------------------------------
    def _worker(self, source_iter):
        try:
            for batch in source_iter:
                # Pin only pageable tensors; already-pinned ones (from
                # DataLoader pin_memory=True) are left as-is.
                pinned = tuple(
                    t.pin_memory()
                    if isinstance(t, torch.Tensor) and not t.is_pinned()
                    else t
                    for t in batch
                )
                # Async DMA on the dedicated transfer stream.
                with torch.cuda.stream(self._stream):
                    on_device = tuple(
                        t.to(self._device, non_blocking=True)
                        if isinstance(t, torch.Tensor)
                        else t
                        for t in pinned
                    )
                event = self._stream.record_event()
                # Queue.put blocks when full → natural back-pressure.
                # Keep *pinned* alive until the consumer waits on *event*;
                # freeing pinned memory while the DMA reads it is UB.
                self._queue.put((on_device, event, pinned))
        except Exception as exc:
            self._queue.put(exc)
        self._queue.put(_PREFETCH_SENTINEL)

    # --- consumer (main / training thread) ------------------------------
    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is _PREFETCH_SENTINEL:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        on_device, event, _pinned = item
        # Block the default (compute) stream until this batch's DMA is done.
        torch.cuda.current_stream(self._device).wait_event(event)
        # _pinned goes out of scope here; safe because wait_event guarantees
        # the DMA has completed on the compute stream's timeline.
        return on_device


class CUDAPrefetcher:
    """Wrap any CPU-batch loader with a VRAM prefetch queue.

    Returns CUDA tensors directly to the training loop.  Existing
    ``.cuda(non_blocking=True)`` calls in the loop become no-ops
    (tensor is already on the target device).

    Works transparently with DataLoader (pin_memory=True or False)
    and PreCollatedBatchStore alike.
    """

    def __init__(self, loader, device, max_prefetch=2):
        self._loader = loader
        self._device = device
        self._max_prefetch = max_prefetch

    def __iter__(self):
        return _VRAMPrefetchIter(
            iter(self._loader), self._device, self._max_prefetch
        )

    def __len__(self):
        return len(self._loader)


class PreCollatedBatchStore:
    """
    Pre-collate all batches in a single pass and store in pageable memory.

    All batches are stored as regular (pageable) CPU tensors.  Pinned memory
    is NOT used for bulk storage — page-locked memory cannot be swapped by
    the OS, so pinning all training data wastes physical RAM and can cause
    system-wide memory pressure.

    transfer_to_device (call AFTER model/optimizer/DDP are on GPU):
        Strategy A -- VRAM fits: move pageable -> VRAM in-place.
        Strategy B -- VRAM insufficient: keep in pageable RAM, use
                      _VRAMPrefetchIter for copy-compute overlap
                      (BG thread: pin → async DMA on a separate stream).

    Drop-in replacement for DataLoader: supports iter(), len(), enumerate().
    Batch composition is fixed at init; batch ORDER is shuffled each epoch.
    """

    def __init__(self, dataset, batch_sampler, collate_fn):
        self.batch_sampler = batch_sampler
        self._batches = []
        self.on_gpu = False
        self._total_bytes = 0
        self._prefetch_device = None
        self._max_prefetch = 2

        all_batch_indices = list(batch_sampler)
        n_batches = len(all_batch_indices)
        logger.info(f"Pre-collating {n_batches} batches...")

        for batch_indices in tqdm(
            all_batch_indices,
            desc="Pre-collating batches",
            file=sys.stdout,
            dynamic_ncols=True,
        ):
            samples = [dataset[i] for i in batch_indices]
            batch = collate_fn(samples)
            self._total_bytes += sum(
                t.nelement() * t.element_size()
                for t in batch
                if isinstance(t, torch.Tensor)
            )
            self._batches.append(batch)

        total_mb = self._total_bytes / (1024 * 1024)
        logger.info(
            f"Pre-collated {n_batches} batches ({total_mb:.0f} MB)"
        )

    # ------------------------------------------------------------------
    # Call AFTER model / optimizer / DDP are on GPU
    # ------------------------------------------------------------------
    def transfer_to_device(self, local_rank, max_vram_usage_ratio=0.45):
        """
        Strategy A: data fits in VRAM → move pageable → VRAM in-place.
        Strategy B: VRAM insufficient → keep pageable, prefetch via
                    _VRAMPrefetchIter (BG thread pin + async DMA).
        """
        free_vram, _ = torch.cuda.mem_get_info(local_rank)
        free_mb = free_vram / (1024 * 1024)
        data_mb = self._total_bytes / (1024 * 1024)
        budget_mb = free_mb * max_vram_usage_ratio

        if data_mb <= budget_mb:
            device = f"cuda:{local_rank}"
            logger.info(
                f"Strategy A (VRAM resident): {len(self._batches)} batches "
                f"({data_mb:.0f} MB) pageable -> VRAM "
                f"(free: {free_mb:.0f} MB, budget: {budget_mb:.0f} MB)"
            )
            # In-place replacement: each pinned batch is freed as soon as
            # its VRAM copy is stored, so RAM shrinks as VRAM grows.
            for i in tqdm(
                range(len(self._batches)),
                desc="-> VRAM",
                file=sys.stdout,
                dynamic_ncols=True,
            ):
                batch = self._batches[i]
                self._batches[i] = tuple(
                    t.to(device, non_blocking=True)
                    if isinstance(t, torch.Tensor)
                    else t
                    for t in batch
                )
            torch.cuda.synchronize(local_rank)
            self.on_gpu = True
            logger.info("All batches are now VRAM-resident")
        else:
            self.on_gpu = False
            self._prefetch_device = f"cuda:{local_rank}"
            # Determine prefetch depth: how many batches to queue on VRAM.
            # Use ~15% of free VRAM for the prefetch buffer, leaving the
            # rest for activations, gradients, and optimizer states.
            avg_batch_mb = data_mb / max(len(self._batches), 1)
            prefetch_vram_mb = free_mb * 0.15
            if avg_batch_mb > 0:
                self._max_prefetch = max(
                    2, min(int(prefetch_vram_mb / avg_batch_mb), len(self._batches))
                )
            else:
                self._max_prefetch = 2
            logger.info(
                f"Strategy B (pageable + VRAM prefetch): "
                f"{len(self._batches)} batches ({data_mb:.0f} MB) in pageable RAM, "
                f"prefetch depth={self._max_prefetch} "
                f"(VRAM free: {free_mb:.0f} MB, budget: {budget_mb:.0f} MB)"
            )

    # ------------------------------------------------------------------
    def shuffle(self, generator=None):
        """Shuffle batch order (call once per epoch)."""
        n = len(self._batches)
        if generator is not None:
            perm = torch.randperm(n, generator=generator).tolist()
        else:
            perm = torch.randperm(n).tolist()
        self._batches = [self._batches[i] for i in perm]

    def __iter__(self):
        if self.on_gpu:
            return iter(self._batches)
        if self._prefetch_device is not None:
            return _VRAMPrefetchIter(
                iter(self._batches),
                self._prefetch_device,
                self._max_prefetch,
            )
        # Fallback: transfer_to_device was not called.
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)
