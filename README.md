# LLM Based ASR Japanese

# Setup
```
uv sync
```

# Data
The audio must be encoded in mp3, you will need to modify data.py to change the format.
```
.
└── ./input/
    ├── /data/
    │   ├── xxx.tar
    │   └── yyy.tar
    └── metadata.csv
```

## metadata.csv
```
__key__,transcription
0eb/016c1fa71688c, XXXX
```

# Example Config
```
TrainerConfig:
  num_train_epochs: 'none'
  max_steps: 10000
  warmup_steps: 1000
  warmup_ratio: none
  learning_rate: 1e-4
  weight_decay: 0.01
  train_batch_size: 4
  eval_batch_size: 4
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  save_strategy: "steps"
  save_steps: 1000

Config:
  url: "path/to/tar files"
  metadata_path: "path/to/csv"
  audio_encoder_name_or_path: "openai/whisper-small"
  text_decoder_name_or_path: "sbintuitions/sarashina2.2-1b-instruct-v0.1"
```

# train
```
python train path/to/config
```

# Acknowledgements
- [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM/tree/main)