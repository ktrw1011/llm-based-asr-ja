from typing import Any

import torch
from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperEncoder, WhisperPreTrainedModel


class WhisperEncoderWrapper(WhisperPreTrainedModel):  # type: ignore[misc]
    def __init__(self, config: WhisperConfig) -> None:
        """_summary_

        Example:
            Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)
        """
        super().__init__(config)
        self.encoder = WhisperEncoder(config)

    def freeze(self) -> None:
        self.encoder._freeze_parameters()

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.encoder(*args, **kwargs) # type: ignore[no-any-return]
