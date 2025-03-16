import os
from pathlib import Path

import peft
import typer
import wandb
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, WhisperFeatureExtractor
from trl import DataCollatorForCompletionOnlyLM

from llm_based_asr_ja.config import Config, TrainerConfig, copy_config
from llm_based_asr_ja.data import AudioTextDataCollator, AudioTextWebDataset
from llm_based_asr_ja.model.modeling_slm import SLMASRConfig, SLMASRForCausalLM
from llm_based_asr_ja.prepare_tokenizer import prepare_tokenizer

load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["WANDB_MODE"] = "online"
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"


def main(config_path: Path) -> None:
    exp_name = config_path.stem
    trainer_cfg = TrainerConfig.load_from_path(config_path)
    cfg = Config.load_from_path(config_path)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.audio_encoder_name_or_path)
    tokenizer, instruction_text, response_template = prepare_tokenizer(cfg.text_decoder_name_or_path)
    audio_tar_paths = cfg.get_tar_path()

    audio_tar_names = [tar_path.stem for tar_path in audio_tar_paths if Path(tar_path).stem != "1b6"]
    audio_tar_path_posix = [tar_path.as_posix() for tar_path in audio_tar_paths if Path(tar_path).stem != "1b6"]

    audio_web_ds = AudioTextWebDataset(
        url=audio_tar_path_posix,
        metadata_path=cfg.metadata_path,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        instruction_text=instruction_text,
    )
    audio_web_ds.metadata = audio_web_ds.metadata.loc[
        [ind for ind in audio_web_ds.metadata.index if ind.split("/")[0] in audio_tar_names]
    ]

    collator = AudioTextDataCollator(
        feature_extractor=feature_extractor,
        completion_only_lm_collator=DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            response_template=response_template,
        ),
    )

    modeling_config = SLMASRConfig(
        audio_encoder_name_or_path=cfg.audio_encoder_name_or_path,
        text_decoder_name_or_path=cfg.text_decoder_name_or_path,
        encoder_projector_ds_rate=5,
    )
    model = SLMASRForCausalLM(config=modeling_config)

    text_decoder_lora_config = LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
    )
    model.text_decoder = get_peft_model(model.text_decoder, text_decoder_lora_config)

    wandb.init(
        project="speech_language_model",
        name=exp_name,
    )

    # config copy to output directory
    output_dir = Path(f"output/{exp_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_config(config_path, output_dir)

    samples_per_step = trainer_cfg.train_batch_size * trainer_cfg.gradient_accumulation_steps
    steps_per_epoch = len(audio_web_ds) // samples_per_step

    if trainer_cfg.max_steps is None:
        if trainer_cfg.num_train_epochs is None:
            raise ValueError("Either max_steps or num_train_epochs must be specified.")
        max_steps = steps_per_epoch * trainer_cfg.num_train_epochs
    else:
        max_steps = trainer_cfg.max_steps

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        log_level="error",
        logging_steps=10,
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=trainer_cfg.save_steps,
        save_total_limit=1,
        max_steps=max_steps,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_steps=trainer_cfg.warmup_steps,
        learning_rate=1e-4,
        weight_decay=trainer_cfg.weight_decay,
        fp16=False,
        bf16=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=False,
        report_to="wandb",
        seed=42,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=audio_web_ds.wds,
    )

    trainer.train()


if __name__ == "__main__":
    typer.run(main)
