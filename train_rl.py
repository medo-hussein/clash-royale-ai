"""
DEEP Q-NETWORK (DQN) AUTONOMOUS TRAINING ENGINE - VERSION 2.0
Handles Mini-Batch Gradient Descent, Bellman Sequence Evaluation, and Gradient Clipping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def optimize_dqn_agent(model, memory, batch_size=64, gamma=0.98, lr=0.0005):
    """خطوة التحديث العصبية والـ Backpropagation للـ Rich State"""
    if len(memory) < batch_size:
        return None

    # سحب عينة عشوائية لكسر الارتباط التراكمي للحركات
    batch = memory.sample(batch_size)
    
    states = torch.FloatTensor(np.array([x[0] for x in batch]))
    actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(1)
    rewards = torch.FloatTensor([x[2] for x in batch])
    next_states = torch.FloatTensor(np.array([x[3] for x in batch]))
    dones = torch.FloatTensor([x[4] for x in batch])

    # حساب قيم الـ Q الحالية
    current_q_values = model(states).gather(1, actions).squeeze(1)

    # حساب الـ Target Q-Values بناءً على معطيات خطوة التنبؤ (Sequence Prediction)
    with torch.no_grad():
        next_q_values = model(next_states).max(1)[0]
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    # حساب الـ Huber Loss لضمان استقرار التدريب ومنع الانحراف المفاجئ
    criterion = nn.HuberLoss()
    loss = criterion(current_q_values, target_q_values)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer.zero_grad()
    loss.backward()
    
    # حماية الأوزان العصبية عبر الـ Gradient Clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()