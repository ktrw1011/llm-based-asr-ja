from typing import Final

from transformers import AutoTokenizer, PreTrainedTokenizer

QWEN_TEMPLATE: Final[str] = "{% for message in messages %}\
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}\
{% endfor %}\
{% if add_generation_prompt %}\
{{ '<|im_start|>assistant\n' }}\
{% endif %}"


def prepare_tokenizer(text_decoder_name_or_path: str) -> tuple[PreTrainedTokenizer, str, str]:
    # set up preprocessing
    tokenizer = AutoTokenizer.from_pretrained(text_decoder_name_or_path)
    if "qwen" in tokenizer.name_or_path.lower():
        tokenizer.pad_token_id = 151654
        response_template = "<|im_start|>assistant"
        instruction_text = "Transcribe the audio clip into text."
        tokenizer.chat_template = QWEN_TEMPLATE  # disable system prompts
    elif "sarashina2.2" in tokenizer.name_or_path.lower():
        tokenizer.pad_token_id = 10
        response_template = "<|assistant|>"
        instruction_text = "音声を文字起こししてください。"
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer.name_or_path}")

    return tokenizer, instruction_text, response_template
