from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from .audio_encoder import WhisperEncoderWrapper
from .projector import EncoderProjectorConcat


def is_peft_model(model: nn.Module) -> bool:
    return hasattr(model, "peft_config")


class SLMASRConfig(PretrainedConfig):  # type: ignore[misc]
    model_type = "slm_asr"

    def __init__(
        self,
        audio_encoder_name_or_path: str = "openai/whisper-small",
        text_decoder_name_or_path: str = "pfnet/plamo-2-1b",
        encoder_projector_ds_rate: int = 5,
        **kwargs: dict[str, Any],
    ) -> None:
        self.audio_encoder_name_or_path = audio_encoder_name_or_path
        self.text_decoder_name_or_path = text_decoder_name_or_path
        self.encoder_projector_ds_rate = encoder_projector_ds_rate

        super().__init__(**kwargs)


class SLMASRForCausalLM(PreTrainedModel):  # type: ignore[misc]
    config_class = SLMASRConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: SLMASRConfig,
    ) -> None:
        super().__init__(config)

        self.audio_encoder = WhisperEncoderWrapper.from_pretrained(
            config.audio_encoder_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.text_decoder = AutoModelForCausalLM.from_pretrained(
            config.text_decoder_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        self.projector = EncoderProjectorConcat(
            encoder_projector_ds_rate=self.config.encoder_projector_ds_rate,
            encoder_dim=self.audio_encoder.config.hidden_size,
            llm_dim=self.text_decoder.config.hidden_size,
        )
        self.audio_encoder_freezed: bool = False
        self.text_decoder_freezed: bool = False

        self._freeze_audio_encoder()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Any | None = None) -> None:
        self.audio_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.text_decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def _freeze_audio_encoder(self) -> None:
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        self.audio_encoder_freezed = True

    def freeze_text_decoder(self) -> None:
        for param in self.text_decoder.parameters():
            param.requires_grad = False
        self.text_decoder_freezed = True

    def save_pretrained(self, save_directory: str | Path, **_kwargs: dict[str, Any]) -> None:
        if isinstance(save_directory, str):
            save_directory = Path(save_directory)

        self.config.save_pretrained(save_directory)
        model_state_dict = {}
        for name, param in self.projector.state_dict().items():
            model_state_dict[f"projector.{name}"] = param
        super().save_pretrained(save_directory, state_dict=model_state_dict)

        if is_peft_model(self.text_decoder) or not self.text_decoder_freezed:
            self.text_decoder.save_pretrained(
                save_directory.joinpath("text_decoder"),
            )

        if is_peft_model(self.audio_encoder) or not self.audio_encoder_freezed:
            self.audio_encoder.save_pretrained(
                save_directory.joinpath("audio_encoder"),
            )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        **kwargs: dict[str, Any],
    ) -> "SLMASRForCausalLM":
        if isinstance(model_name_or_path, str):
            model_name_or_path = Path(model_name_or_path)
        config = SLMASRConfig.from_pretrained(model_name_or_path, **kwargs)
        model = cls(config)

        if model_name_or_path.joinpath("audio_encoder").exists():
            model.audio_encoder = WhisperEncoderWrapper.from_pretrained(
                model_name_or_path,
                subfolder="audio_encoder",
                **kwargs,
            )
        if model_name_or_path.joinpath("text_decoder/adapter_config.json").exists():
            # Load adapter config
            from peft import PeftModel

            model.text_decoder = PeftModel.from_pretrained(
                model.text_decoder,
                model_name_or_path.joinpath("text_decoder"),
            )
            model.text_decoder = model.text_decoder.merge_and_unload()
        elif model_name_or_path.joinpath("text_decoder").exists():
            model.text_decoder = AutoModelForCausalLM.from_pretrained(
                model_name_or_path.joinpath("text_decoder"),
                **kwargs,
            )

        state_dict = {
            k.replace("projector.", ""): v
            for k, v in load_file(
                model_name_or_path.joinpath("model.safetensors"),
            ).items()
        }
        model.projector.load_state_dict(state_dict)

        return model

    def forward(
        self,
        input_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ) -> CausalLMOutputWithPast:
        text_decoder = self.text_decoder.model if is_peft_model(self.text_decoder) else self.text_decoder

        audio_embed = self.audio_encoder(input_features=input_features)
        prj_out = self.projector(audio_embed.last_hidden_state)
        llm_embed = text_decoder.model.embed_tokens(input_ids)

        concat_embed = torch.cat((prj_out, llm_embed), dim=1)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = text_decoder.model(inputs_embeds=concat_embed)
        hidden_states = outputs[0]

        logits = text_decoder.lm_head(hidden_states[:, -input_ids.size(1) :, :])

        loss = None
        if labels is not None:
            loss = text_decoder.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=text_decoder.config.vocab_size,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


# AutoConfig.register("slm_asr", SLMASRConfig)
# AutoModelForCausalLM.register(SLMASRConfig, SLMASRForCausalLM)
# SLMASRConfig.register_for_auto_class()
# SLMASRForCausalLM.register_for_auto_class("AutoModelForCausalLM")
