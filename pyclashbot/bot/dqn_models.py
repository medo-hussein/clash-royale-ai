"""
GRANDMASTER UNIVERSAL RL BRAIN - VERSION 2.0 (RICH GAME STATE CORE)
Features: 220-Dimensional State Input, Advanced Policy-Action Matrix Output,
          and Deep Sequential Replay Memory Buffer.
"""

import torch
import torch.nn as nn
import collections
import random
import numpy as np

class DQNCrashBrain(nn.Module):
    def __init__(self, input_dim=220, output_dim=16): 
        """
        Input: 220 عنصر يمثلون كامل تفاصيل المعركة والذاكرة التراكمية.
        Output: 16 مخرجاً لتقييم مصفوفة القرار (4 كروت يد × 4 مناطق جغرافية رئيسية بالملعب).
        """
        super(DQNCrashBrain, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        # طبقة حساب القيمة النفعية لكل قرار (Q-Value Matrix)
        self.q_value_head = nn.Linear(256, output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.q_value_head(features)

class TransitionBuffer:
    def __init__(self, capacity=20000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)