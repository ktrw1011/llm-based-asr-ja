# This file contains code copied from https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/models/projector.py
# Copyright (c) 2024 Ziyang Ma
# Licensed under the MIT License

import torch
from torch import nn


class EncoderProjectorConcat(nn.Module):
    def __init__(self, encoder_projector_ds_rate: int, encoder_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.k = encoder_projector_ds_rate
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class EncoderProjectorCov1d(nn.Module):
    def __init__(self, encoder_projector_ds_rate: int, encoder_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.k = encoder_projector_ds_rate
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.encoder_dim,
            out_channels=self.encoder_dim,
            kernel_size=self.k,
            stride=self.k,
            padding=0,
        )
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x
