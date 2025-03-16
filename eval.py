from pathlib import Path
from typing import cast

import jiwer
import numpy as np
import pandas as pd
import torch
import typer
import webdataset as wds
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, WhisperFeatureExtractor

from llm_based_asr_ja.config import Config
from llm_based_asr_ja.data import load_bytes_audio
from llm_based_asr_ja.model.modeling_slm import SLMASRForCausalLM
from llm_based_asr_ja.text_normalize import normalize_text


def get_prefix(model_name_or_path: str) -> str:
    if "qwen" in model_name_or_path.lower():
        return "<|im_end|>\n<|im_start|>user\nTranscribe the audio clip into text.<|im_end|>\n<|im_start|>assistant"
    if "sarashina" in model_name_or_path.lower():
        return "<|user|>音声を文字起こししてください。</s><|assistant|>"
    raise ValueError(f"Unknown model name: {model_name_or_path}")


def compute_cer(preds: list[str], labels: list[str]) -> float:
    preds = [normalize_text(pred) for pred in preds]
    labels = [normalize_text(label) for label in labels]

    cer = jiwer.cer(reference=labels, hypothesis=preds)
    return cast("float", cer)


def get_latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError
    return max(checkpoints, key=lambda x: int(x.stem.split("-")[-1]))


@torch.inference_mode()
def generate(
    weveform: np.ndarray,
    model: SLMASRForCausalLM,
    tokenizer: PreTrainedTokenizer,
    feature_extractor: WhisperFeatureExtractor,
    prefix: str,
) -> str:
    model_dtype = next(model.parameters()).dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mel_feature = feature_extractor(
        weveform,
        return_tensors="pt",
        sampling_rate=feature_extractor.sampling_rate,
    ).input_features

    with torch.amp.autocast(dtype=model_dtype, device_type=device):
        audio_embed = model.audio_encoder(input_features=mel_feature.to(device))
        prj_out = model.projector(audio_embed.last_hidden_state)

        llm_embed = model.text_decoder.model.embed_tokens(tokenizer.encode(prefix, return_tensors="pt").to(device))
        concat_embed = torch.cat((prj_out, llm_embed), dim=1)
        outputs = model.text_decoder.generate(
            inputs_embeds=concat_embed,
            max_new_tokens=256,
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return str(output_text)


def main(model_path: Path, test_tar_path: Path, metadata_csv_path: Path) -> None:
    audio_web_ds = wds.WebDataset(test_tar_path.as_posix())
    typer.echo(f"test tar: {test_tar_path}")

    cfg = Config.load_from_path(model_path.joinpath("config.yaml"))
    typer.echo(f"config: {cfg}")

    df = pd.read_csv(metadata_csv_path, index_col=0)
    df = df.loc[[ind for ind in df.index if ind.split("/")[0] == test_tar_path.stem]]

    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.audio_encoder_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_decoder_name_or_path)

    model = SLMASRForCausalLM.from_pretrained(get_latest_checkpoint(model_path))

    model.eval()
    model.to("cuda")
    model.text_decoder.generation_config.pad_token_id = tokenizer.pad_token_id

    prefix = get_prefix(cfg.text_decoder_name_or_path)
    preds: list[str] = []
    labels: list[str] = []
    for sample in tqdm(audio_web_ds, total=len(df)):
        waveform = load_bytes_audio(sample["mp3"], target_sampling_rate=feature_extractor.sampling_rate)
        pred = generate(waveform, model, tokenizer, feature_extractor, prefix)
        preds.append(pred)
        labels.append(df.loc[sample["__key__"]]["pred"])

    cer_score = compute_cer(preds, labels)
    typer.echo(f"CER: {cer_score:.3f}")


if __name__ == "__main__":
    typer.run(main)
