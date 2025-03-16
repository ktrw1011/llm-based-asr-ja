import io
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from transformers import PreTrainedTokenizer, WhisperFeatureExtractor
from transformers.data.data_collator import DataCollatorMixin
from trl import DataCollatorForCompletionOnlyLM

from llm_based_asr_ja.text_normalize import delete_symbols, normalize_text

MAX_SECONDS = 30.0


def is_inseconds(row: dict) -> bool:
    audio = row["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    return bool(duration <= MAX_SECONDS)


def lower_text(row: dict) -> dict:
    text: str = row["text"]
    row["text"] = text.lower()
    return row


def load_bytes_audio(value: bytes, target_sampling_rate: int) -> np.ndarray:
    waveform, sr = librosa.load(io.BytesIO(value), mono=True)

    if sr != target_sampling_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sampling_rate)
    return waveform


class AudioTextWebDataset:
    def __init__(  # noqa: PLR0913
        self,
        url: str | list[str],
        metadata_path: str,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: PreTrainedTokenizer,
        instruction_text: str,
        down_sampling_k: int = 5,
    ) -> None:
        self.url = url
        self.metadata = pd.read_csv(metadata_path, index_col=0)

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.instruction_text = instruction_text
        self.down_sampling_k = down_sampling_k
        self.sampling_rate = feature_extractor.sampling_rate

        self._init_wds()

    def __len__(self) -> int:
        return len(self.metadata)

    def _init_wds(self) -> None:
        self.wds = wds.WebDataset(self.url).map(self._process_sample)

    def _process_sample(self, sample: dict[str, Any]) -> dict[str, list[int] | torch.Tensor]:
        key: str = sample["__key__"]
        waveform = load_bytes_audio(sample["mp3"], self.sampling_rate)

        # Extract features from audio
        # (mel_feature, frame_length)
        mel_feature = self.feature_extractor(
            waveform,
            return_tensors="pt",
            sampling_rate=self.sampling_rate,
        ).input_features[0]

        # audio_frame_len = mel_feature.shape[-1]
        # audio_frame_len = (audio_frame_len + 1) // 2  # down sample by whisper (2)
        # audio_frame_len = audio_frame_len // self.down_sampling_k  # down sample by projector (k)

        # Tokenize text
        text: str = self.metadata.loc[key, "transcription"]
        text_encoded = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": self.instruction_text},
                {"role": "assistant", "content": delete_symbols(normalize_text(text))},
            ],
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = text_encoded["input_ids"][0]

        # attention_mask = text_encoded["attention_mask"][0]

        # extend attention mask
        # audio_mask = torch.full((audio_frame_len,), -1)
        # concat_mask = torch.cat((audio_mask, attention_mask))
        # extended_attention_mask = concat_mask.ge(-1)

        return {
            "input_features": mel_feature,
            "input_ids": input_ids,
            # "attention_mask": extended_attention_mask,
        }


@dataclass
class AudioTextDataCollator(DataCollatorMixin):  # type: ignore[misc]
    feature_extractor: WhisperFeatureExtractor
    completion_only_lm_collator: DataCollatorForCompletionOnlyLM
    return_tensors: str = "pt"

    def torch_call(self, features: list[dict[str, list[int] | torch.Tensor]]) -> dict[str, Any]:
        batch = {}
        # audio features
        # whisper already does padding at the feature extractor
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        audio_batch = self.feature_extractor.pad(input_features, return_tensors=self.return_tensors)

        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        text_batch = self.completion_only_lm_collator(input_ids, return_tensors=self.return_tensors)

        batch["input_features"] = audio_batch["input_features"]
        batch["input_ids"] = text_batch["input_ids"]
        batch["labels"] = text_batch["labels"]

        return batch
