"""
DRAGON ULTIMATE V25
===================================================================================
Fully merged production-ready Clash Royale AI.

Architecture:
  V24 ObjectiveDriven placement as the backbone.
  V23 MetaKnowledgeBase, PsychWarfareEngine, AdvancedVisionSystem, TowerDamageTracker,
      split-push, freeze-combo, king-activation, precision placements — all carried in.
  V23 9 critical bug-fixes + V22 fixgroup — all applied.
  NEW: EnemyRotationTracker V25 API (cards_until / predict_next_card / is_out_of_cycle)
  NEW: Enhanced PredictionEngine using cycle + archetype + elixir + board.
  NEW: ThreatHeatmap with named danger_map / pressure_map / spell_value_map.
  NEW: StrategicBrain multi-step planning with board-reactive plan advancement.
  NEW: MatchAnalytics with full per-match statistics.
  NEW: TerminalDashboard with text-based trend charts.

V23 Bug-Fixes Applied
---------------------
  FIX 1: target_dqn updates every TARGET_UPDATE_FREQ=1000 steps ONLY.
  FIX 2: No Q-value boosting (DQN learns counters through reward shaping).
  FIX 3: ONLINE_TRAIN_FREQ=4 (was 15).
  FIX 4: NStepBuffer deque without maxlen; popleft() manually after extraction.
  FIX 5: Terminal reward graduated by towers destroyed / lost.
  FIX 6: Epsilon decays only on real card actions, not on WAIT.
  FIX 7: Model checkpoint every 200 steps (not only after match).
  FIX 8: PER buffer capacity 50 000.
  FIX 9: Win-rate CSV logged after every match.
  FIX A: map_coordinates pass-through with COORD_MIRROR flags.
  FIX B: State vector 250-dim including 4 hand-card slots.
  FIX C: HP bar detector multi-colour (green + yellow + red).
  FIX D: Thread-safe tracker snapshot in extract_state.
  FIX E: Adaptive frame timing (120 ms/frame target).
  FIX F: Available-indices mask strict across all 37 actions.
===================================================================================
"""

import collections
import random
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import threading
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from pyclashbot.detection.yolo_vision import get_yolo_predictions
from pyclashbot.bot.card_detection import check_which_cards_are_available, YOLO_TO_BOT_NAMES
from pyclashbot.bot.constants import CLASH_MAIN_DEADSPACE_COORD
from pyclashbot.bot.nav import check_if_in_battle, check_if_on_clash_main_menu, wait_for_battle_start
from pyclashbot.detection.image_rec import find_image, pixel_is_equal
from pyclashbot.utils.cancellation import interruptible_sleep
from pyclashbot.utils.logger import Logger


# =============================================================================
# CONFIGURATION & COORDINATE SYSTEM
# =============================================================================

# [FIX A] Pass-through with optional mirror — change flags if emulator is flipped
COORD_MIRROR_X: bool = False
COORD_MIRROR_Y: bool = False
COORD_X_MIN, COORD_X_MAX = 20, 340
COORD_Y_MIN, COORD_Y_MAX = 50, 570


def map_coordinates(x: int, y: int) -> Tuple[int, int]:
    fx = (COORD_X_MAX + COORD_X_MIN - x) if COORD_MIRROR_X else x
    fy = (COORD_Y_MAX + COORD_Y_MIN - y) if COORD_MIRROR_Y else y
    return (int(max(COORD_X_MIN, min(COORD_X_MAX, fx))),
            int(max(COORD_Y_MIN, min(COORD_Y_MAX, fy))))


ENEMY_TOWER_L_RAW     = (115, 160)
ENEMY_TOWER_R_RAW     = (290, 160)
ENEMY_KING_TOWER_RAW  = (200, 110)
MY_TOWER_L_RAW        = (115, 455)
MY_TOWER_R_RAW        = (290, 455)
MY_KING_TOWER_RAW     = (200, 510)
HAND_CARDS_COORDS     = [(142, 561), (210, 563), (272, 561), (341, 563)]
BRIDGE_LEFT_ATTACK    = (115, 310)
BRIDGE_RIGHT_ATTACK   = (290, 310)

SPATIAL_ZONES_9: List[Tuple[int, int]] = []
ZONE_XS = [60, 87, 114, 141, 168, 200, 228, 255, 282]
ZONE_YS = [420, 360, 300, 230]
for _zy in ZONE_YS:
    for _zx in ZONE_XS:
        SPATIAL_ZONES_9.append((_zx, _zy))
assert len(SPATIAL_ZONES_9) == 36

DEFENSE_ZONES: Dict[str, List[Tuple[int, int]]] = {
    "left":   [(90, 420), (115, 400), (140, 380)],
    "right":  [(255, 420), (290, 400), (315, 380)],
    "center": [(200, 410), (200, 390), (200, 370)],
}

PRECISION_PLACEMENTS: Dict[str, Tuple[int, int]] = {
    "king_activation_left":  (62, 390),
    "king_activation_right": (298, 390),
    "cannon_vs_hog_left":    (115, 430),
    "cannon_vs_hog_right":   (245, 430),
    "kite_left_deep":        (60, 350),
    "kite_right_deep":       (300, 350),
    "funnel_left":           (80, 290),
    "funnel_right":          (280, 290),
    "spell_behind_tower_l":  (90, 130),
    "spell_behind_tower_r":  (270, 130),
    "graveyard_vs_tower_l":  (105, 155),
    "graveyard_vs_tower_r":  (255, 155),
    "barrel_corner_l":       (68, 175),
    "barrel_corner_r":       (292, 175),
}

# [FIX 8] Buffer capacity 50 000
MODEL_FILE       = "clash_dqn_v25.pth"
EPSILON_FILE     = "clash_epsilon_v25.txt"
MATCH_STATS_FILE = "clash_match_stats_v25.csv"
TARGET_UPDATE_FREQ = 1000   # [FIX 1]
ONLINE_TRAIN_FREQ  = 4      # [FIX 3]
ONLINE_BATCH_SIZE  = 32
STATE_DIM          = 250    # [FIX B]
ACTION_DIM         = 37
WAIT_ACTION_INDEX  = 36


# =============================================================================
# CARD PROFILES & ARCHETYPE DATA
# =============================================================================

CARD_PROFILES: Dict[str, Dict] = {
    # Win Conditions
    "hog":           {"role": "WIN_CONDITION", "cost": 4, "mechanic": "BRIDGE_SPAM",    "targets": "building", "air": False},
    "hog rider":     {"role": "WIN_CONDITION", "cost": 4, "mechanic": "BRIDGE_SPAM",    "targets": "building", "air": False},
    "balloon":       {"role": "WIN_CONDITION", "cost": 5, "mechanic": "AIR_PUSH",       "targets": "building", "air": True},
    "royal giant":   {"role": "WIN_CONDITION", "cost": 6, "mechanic": "BRIDGE_SPAM",    "targets": "building", "air": False},
    "xbow":          {"role": "WIN_CONDITION", "cost": 6, "mechanic": "SIEGE",          "targets": "building", "air": False},
    "mortar":        {"role": "WIN_CONDITION", "cost": 4, "mechanic": "SIEGE",          "targets": "building", "air": False},
    "miner":         {"role": "WIN_CONDITION", "cost": 3, "mechanic": "DIRECT_TOWER",   "targets": "building", "air": False},
    "goblin_barrel": {"role": "WIN_CONDITION", "cost": 3, "mechanic": "DIRECT_TOWER",   "targets": "any",      "air": False},
    "goblin barrel": {"role": "WIN_CONDITION", "cost": 3, "mechanic": "DIRECT_TOWER",   "targets": "any",      "air": False},
    "ram_rider":     {"role": "WIN_CONDITION", "cost": 5, "mechanic": "BRIDGE_SPAM",    "targets": "building", "air": False},
    "ram rider":     {"role": "WIN_CONDITION", "cost": 5, "mechanic": "BRIDGE_SPAM",    "targets": "building", "air": False},
    "graveyard":     {"role": "WIN_CONDITION", "cost": 5, "mechanic": "DIRECT_TOWER",   "targets": "any",      "air": False},
    # Tanks
    "giant":         {"role": "TANK",          "cost": 5, "mechanic": "SLOW_BUILD",     "targets": "building", "air": False},
    "golem":         {"role": "TANK",          "cost": 8, "mechanic": "SLOW_BUILD",     "targets": "building", "air": False},
    "giant_skeleton":{"role": "TANK",          "cost": 6, "mechanic": "SLOW_BUILD",     "targets": "any",      "air": False},
    "lava_hound":    {"role": "TANK",          "cost": 7, "mechanic": "AIR_PUSH",       "targets": "building", "air": True},
    # Tank Killers
    "pekka":         {"role": "TANK_KILLER",   "cost": 7, "mechanic": "COUNTER_PUSH",   "targets": "ground",   "air": False},
    "mini_pekka":    {"role": "TANK_KILLER",   "cost": 4, "mechanic": "FRONT_INTERCEPT","targets": "ground",   "air": False},
    "mini p.e.k.k.a":{"role": "TANK_KILLER",  "cost": 4, "mechanic": "FRONT_INTERCEPT","targets": "ground",   "air": False},
    "prince":        {"role": "TANK_KILLER",   "cost": 5, "mechanic": "FRONT_INTERCEPT","targets": "ground",   "air": False},
    "dark_prince":   {"role": "TANK_KILLER",   "cost": 4, "mechanic": "FRONT_INTERCEPT","targets": "ground",   "air": False},
    "sparky":        {"role": "TANK_KILLER",   "cost": 6, "mechanic": "SAFE_SUPPORT",   "targets": "ground",   "air": False},
    # Air Defense
    "musketeer":     {"role": "AIR_DEFENSE",   "cost": 4, "mechanic": "SAFE_SUPPORT",   "targets": "any",      "air": False},
    "minions":       {"role": "AIR_DEFENSE",   "cost": 3, "mechanic": "SWARM",          "targets": "any",      "air": True},
    "minion_horde":  {"role": "AIR_DEFENSE",   "cost": 5, "mechanic": "SWARM",          "targets": "any",      "air": True},
    "minion horde":  {"role": "AIR_DEFENSE",   "cost": 5, "mechanic": "SWARM",          "targets": "any",      "air": True},
    "mega_minion":   {"role": "AIR_DEFENSE",   "cost": 3, "mechanic": "SAFE_SUPPORT",   "targets": "any",      "air": True},
    "inferno_dragon":{"role": "AIR_DEFENSE",   "cost": 4, "mechanic": "MELT",           "targets": "any",      "air": True},
    "electro_dragon":{"role": "AIR_DEFENSE",   "cost": 5, "mechanic": "CHAIN_STUN",     "targets": "any",      "air": True},
    # Support
    "witch":         {"role": "SUPPORT",       "cost": 5, "mechanic": "SAFE_SUPPORT",   "targets": "any",      "air": False},
    "wizard":        {"role": "SUPPORT",       "cost": 5, "mechanic": "SAFE_SUPPORT",   "targets": "any",      "air": False},
    "baby_dragon":   {"role": "SUPPORT",       "cost": 4, "mechanic": "SAFE_SUPPORT",   "targets": "any",      "air": True},
    "electro_wizard":{"role": "SUPPORT",       "cost": 4, "mechanic": "RESET_STUN",     "targets": "any",      "air": False},
    "night_witch":   {"role": "SUPPORT",       "cost": 4, "mechanic": "SAFE_SUPPORT",   "targets": "ground",   "air": False},
    "hunter":        {"role": "SUPPORT",       "cost": 4, "mechanic": "SAFE_SUPPORT",   "targets": "ground",   "air": False},
    # Buildings
    "cannon":        {"role": "BUILDING",      "cost": 3, "mechanic": "CENTER_PULL",    "targets": "ground",   "air": False},
    "tesla":         {"role": "BUILDING",      "cost": 4, "mechanic": "CENTER_PULL",    "targets": "any",      "air": False},
    "inferno_tower": {"role": "BUILDING",      "cost": 5, "mechanic": "MELT",           "targets": "ground",   "air": False},
    "bomb_tower":    {"role": "BUILDING",      "cost": 4, "mechanic": "CENTER_PULL",    "targets": "ground",   "air": False},
    "goblin_cage":   {"role": "BUILDING",      "cost": 4, "mechanic": "CENTER_PULL",    "targets": "ground",   "air": False},
    # Cycle
    "skeletons":     {"role": "CYCLE",         "cost": 1, "mechanic": "SWARM_DISTRACTION","targets": "ground",  "air": False},
    "ice_spirit":    {"role": "CYCLE",         "cost": 1, "mechanic": "FREEZE_STALL",   "targets": "any",      "air": False},
    "fire_spirit":   {"role": "CYCLE",         "cost": 1, "mechanic": "SWARM_CLEAR",    "targets": "any",      "air": False},
    "bats":          {"role": "CYCLE",         "cost": 2, "mechanic": "SWARM_DISTRACTION","targets": "any",    "air": True},
    "goblins":       {"role": "CYCLE",         "cost": 2, "mechanic": "SWARM",          "targets": "ground",   "air": False},
    "ice_golem":     {"role": "MINI_TANK",     "cost": 2, "mechanic": "KITING",         "targets": "building", "air": False},
    "goblin_gang":   {"role": "CYCLE",         "cost": 3, "mechanic": "SWARM",          "targets": "ground",   "air": False},
    "goblin gang":   {"role": "CYCLE",         "cost": 3, "mechanic": "SWARM",          "targets": "ground",   "air": False},
    # Spells
    "fireball":      {"role": "SPELL_HEAVY",   "cost": 4, "mechanic": "VALUE_CLUSTER",  "targets": "any",      "air": False},
    "rocket":        {"role": "SPELL_HEAVY",   "cost": 6, "mechanic": "TOWER_SNIPE",    "targets": "any",      "air": False},
    "lightning":     {"role": "SPELL_HEAVY",   "cost": 6, "mechanic": "VALUE_CLUSTER",  "targets": "any",      "air": False},
    "zap":           {"role": "SPELL_LIGHT",   "cost": 2, "mechanic": "RESET_CLEAR",    "targets": "any",      "air": False},
    "arrows":        {"role": "SPELL_LIGHT",   "cost": 3, "mechanic": "RESET_CLEAR",    "targets": "any",      "air": False},
    "log":           {"role": "SPELL_LIGHT",   "cost": 2, "mechanic": "RESET_CLEAR",    "targets": "ground",   "air": False},
    "earthquake":    {"role": "SPELL_MEDIUM",  "cost": 3, "mechanic": "BUILDING_DAMAGE","targets": "ground",   "air": False},
    "freeze":        {"role": "SPELL_HEAVY",   "cost": 4, "mechanic": "FREEZE_PUSH",    "targets": "any",      "air": False},
    "poison":        {"role": "SPELL_HEAVY",   "cost": 4, "mechanic": "ZONE_CONTROL",   "targets": "any",      "air": False},
}

ROLE_THREAT_WEIGHTS: Dict[str, int] = {
    "WIN_CONDITION": 10, "TANK": 9, "TANK_KILLER": 8,
    "AIR_DEFENSE": 7, "SUPPORT": 6, "BUILDING": 5,
    "MINI_TANK": 4, "CYCLE": 2,
    "SPELL_HEAVY": 0, "SPELL_LIGHT": 0, "SPELL_MEDIUM": 0,
}

ROLE_TO_ID: Dict[str, float] = {
    "WIN_CONDITION": 0.9, "TANK": 0.8, "TANK_KILLER": 0.75,
    "AIR_DEFENSE": 0.65, "SUPPORT": 0.55, "BUILDING": 0.5,
    "MINI_TANK": 0.4, "CYCLE": 0.25,
    "SPELL_HEAVY": 0.7, "SPELL_MEDIUM": 0.6, "SPELL_LIGHT": 0.3,
    "UNKNOWN": 0.1,
}

MECHANIC_TO_ID: Dict[str, float] = {
    "BRIDGE_SPAM": 0.9, "SLOW_BUILD": 0.8, "AIR_PUSH": 0.75, "SIEGE": 0.7,
    "DIRECT_TOWER": 0.65, "FRONT_INTERCEPT": 0.6, "SAFE_SUPPORT": 0.5, "MELT": 0.45,
    "CHAIN_STUN": 0.4, "RESET_STUN": 0.35, "CENTER_PULL": 0.3, "SWARM": 0.25,
    "SWARM_DISTRACTION": 0.2, "KITING": 0.15, "VALUE_CLUSTER": 0.55,
    "FREEZE_STALL": 0.35, "SWARM_CLEAR": 0.25, "FREEZE_PUSH": 0.45,
    "ZONE_CONTROL": 0.5, "RESET_CLEAR": 0.3, "BUILDING_DAMAGE": 0.35,
    "TOWER_SNIPE": 0.8, "COUNTER_PUSH": 0.6, "UNKNOWN": 0.1,
}

ARCHETYPE_COUNTERS: Dict[str, Dict] = {
    "HOG_2.6_CYCLE": {
        "best_counters": ["cannon", "tesla", "bomb_tower", "inferno_tower", "mini_pekka", "goblin_cage"],
        "spell_target": "cluster", "play_style": "CYCLE_DEFENSE",
        "avoid_overcommit": True, "counter_push": True,
    },
    "BEATDOWN_PUSH": {
        "best_counters": ["inferno_tower", "inferno_dragon", "pekka", "mini_pekka", "sparky", "poison"],
        "spell_target": "tank", "play_style": "HARD_DEFENSE",
        "avoid_overcommit": False, "counter_push": True,
    },
    "LOG_BAIT": {
        "best_counters": ["arrows", "zap", "fireball", "log", "electro_wizard", "minion_horde"],
        "spell_target": "swarm", "play_style": "SPELL_DISCIPLINE",
        "avoid_overcommit": True, "counter_push": False,
    },
    "SIEGE": {
        "best_counters": ["miner", "goblin_barrel", "hog rider", "ram_rider", "rocket"],
        "spell_target": "tower_snipe", "play_style": "RUSH_PRESSURE",
        "avoid_overcommit": False, "counter_push": True,
    },
    "BRIDGE_SPAM": {
        "best_counters": ["pekka", "mini_pekka", "inferno_tower", "cannon", "fireball"],
        "spell_target": "cluster", "play_style": "HARD_DEFENSE",
        "avoid_overcommit": False, "counter_push": True,
    },
    "GRAVEYARD_CONTROL": {
        "best_counters": ["tesla", "bomb_tower", "arrows", "poison"],
        "spell_target": "graveyard_activation", "play_style": "BUILDING_PLACEMENT",
        "avoid_overcommit": True, "counter_push": False,
    },
    "MINER_CONTROL": {
        "best_counters": ["tesla", "cannon", "inferno_tower", "musketeer", "bomb_tower"],
        "spell_target": "value_cluster", "play_style": "BUILDING_PLACEMENT",
        "avoid_overcommit": True, "counter_push": False,
    },
    "LAVALOON": {
        "best_counters": ["inferno_tower", "musketeer", "electro_dragon", "tesla"],
        "spell_target": "cluster", "play_style": "ANTI_AIR",
        "avoid_overcommit": False, "counter_push": True,
    },
    "UNKNOWN": {
        "best_counters": [], "spell_target": "cluster",
        "play_style": "BALANCED", "avoid_overcommit": False, "counter_push": True,
    },
}

# Update YOLO name map
YOLO_TO_BOT_NAMES.update({
    "Mini_Pekka_Card": "mini_pekka", "Mini_Pekka": "mini_pekka",
    "Tesla_Card": "tesla", "Tesla": "tesla",
    "Royal_Giant_Card": "royal giant", "Fireball_Card": "fireball",
    "Fireball": "fireball", "Zap_Card": "zap", "Zap": "zap",
    "Arrows_Card": "arrows", "Arrows": "arrows",
    "Goblin_Barrel_Card": "goblin_barrel", "Goblin_Barrel": "goblin_barrel",
    "Hog_Rider_Card": "hog rider", "Hog_Rider": "hog rider",
    "Ram_Rider_Card": "ram_rider", "Balloon_Card": "balloon",
    "Balloon": "balloon", "Golem_Card": "golem",
    "Giant_Card": "giant", "Pekka_Card": "pekka",
    "Freeze_Card": "freeze", "Log_Card": "log", "Log": "log",
    "Rocket_Card": "rocket", "Rocket": "rocket",
    "Poison_Card": "poison", "Poison": "poison",
    "Prince_Card": "prince", "Sparky_Card": "sparky",
    "Inferno_Tower_Card": "inferno_tower",
    "Lightning_Card": "lightning", "Lightning": "lightning",
})


# =============================================================================
# NEURAL NETWORK — DUELING DOUBLE DQN
# =============================================================================

class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_net(x)
        values = self.value_stream(feats)
        advs = self.advantage_stream(feats)
        return values + (advs - advs.mean(dim=1, keepdim=True))


class DQNCrashBrain:
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, lr: float = 0.0001):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model      = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target     = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.optimizer  = optim.Adam(self.model.parameters(), lr=lr)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def load_state_dict(self, sd: dict) -> None:
        self.model.load_state_dict(sd)
        self.target.load_state_dict(sd)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def eval(self) -> None:
        self.model.eval()

    def train(self) -> None:
        self.model.train()


def optimize_dqn_agent(agent: DQNCrashBrain, replay_buffer, batch_size: int = 32,
                        gamma: float = 0.99, target_net: Optional[DQNCrashBrain] = None) -> Optional[float]:
    if len(replay_buffer) < batch_size:
        return None

    if hasattr(replay_buffer, "sample"):
        samples, indices, weights = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        weights = torch.FloatTensor(weights).to(agent.device)
    else:
        batch = random.sample(replay_buffer.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        indices = None
        weights = torch.ones(batch_size, device=agent.device)

    states      = torch.FloatTensor(np.array(states)).to(agent.device)
    actions     = torch.LongTensor(actions).to(agent.device)
    rewards     = torch.FloatTensor(rewards).to(agent.device)
    next_states = torch.FloatTensor(np.array(next_states)).to(agent.device)
    dones       = torch.FloatTensor(dones).to(agent.device)

    q_values  = agent.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    t_net     = target_net if target_net else agent
    with torch.no_grad():
        next_actions = agent.model(next_states).argmax(dim=1)
        target_q     = t_net.model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q     = rewards + gamma * target_q * (1 - dones)

    loss = (weights * (q_values - target_q).pow(2)).mean()
    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 10.0)
    agent.optimizer.step()

    if indices is not None and hasattr(replay_buffer, "update_priorities"):
        td_errors = (q_values - target_q).detach().cpu().numpy()
        replay_buffer.update_priorities(indices, td_errors)

    return loss.item()


# =============================================================================
# MEMORY SYSTEMS
# =============================================================================

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay. [FIX 8] Capacity = 50 000."""

    def __init__(self, capacity: int = 50000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity     = capacity
        self.alpha        = alpha
        self.beta         = beta
        self.buffer: list = []
        self.priorities   = []
        self.pos          = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done) -> None:
        exp = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
            self.priorities.append(self.max_priority)
        else:
            self.buffer[self.pos]    = exp
            self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[list, np.ndarray, np.ndarray]:
        if not self.buffer:
            return [], np.array([]), np.ones(0)
        probs   = np.array(self.priorities[:len(self.buffer)]) ** self.alpha
        probs  /= probs.sum()
        actual  = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), actual, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, td_errors) -> None:
        for idx, td in zip(indices, td_errors):
            p = (abs(td) + 1e-6) ** self.alpha
            self.priorities[idx] = p
            self.max_priority = max(self.max_priority, p)

    def __len__(self) -> int:
        return len(self.buffer)


class NStepBuffer:
    """N-step Returns Buffer. [FIX 4] deque without maxlen, manual popleft."""

    def __init__(self, n: int = 5, gamma: float = 0.99):
        self.n     = n
        self.gamma = gamma
        self.buf: collections.deque = collections.deque()

    def push(self, state, action, reward: float, next_state, done: bool) -> None:
        self.buf.append((state, action, reward, next_state, done))

    def get_nstep(self) -> Optional[tuple]:
        if len(self.buf) < self.n:
            return None
        state, action, _, _, _ = self.buf[0]
        G = 0.0
        for i, (_, _, r, _, d) in enumerate(self.buf):
            G += (self.gamma ** i) * r
            if d:
                break
        last_next = self.buf[-1][3]
        last_done = self.buf[-1][4]
        self.buf.popleft()
        return (state, action, G, last_next, last_done)

    def flush(self) -> List[tuple]:
        results = []
        while self.buf:
            state, action, _, _, _ = self.buf[0]
            G = 0.0
            for i, (_, _, r, _, d) in enumerate(self.buf):
                G += (self.gamma ** i) * r
                if d:
                    break
            last_next = self.buf[-1][3]
            last_done = self.buf[-1][4]
            results.append((state, action, G, last_next, last_done))
            self.buf.popleft()
        return results

    def clear(self) -> None:
        self.buf.clear()


# =============================================================================
# ENEMY ROTATION TRACKER  (V25 interface)
# =============================================================================

class EnemyRotationTracker:
    """
    Tracks the full 8-card enemy deck, cycle position, and upcoming threats.
    V25 interface: cards_until() / predict_next_card() / is_out_of_cycle()
    """

    FULL_DECK_SIZE  = 8
    HAND_SIZE       = 4
    WIN_CONDS       = {
        "hog rider", "hog", "balloon", "golem", "giant", "royal giant",
        "xbow", "mortar", "ram_rider", "ram rider", "graveyard", "goblin barrel",
        "goblin_barrel", "miner", "lava_hound",
    }
    SPELLS          = {"fireball", "rocket", "lightning", "zap", "arrows", "log",
                       "freeze", "poison", "earthquake"}
    BUILDINGS       = {"cannon", "tesla", "inferno_tower", "bomb_tower", "goblin_cage"}

    def __init__(self):
        self.enemy_deck:    List[str]  = []
        self.last_seen:     Dict[str, float] = {}
        self.play_count:    Dict[str, int]   = {}
        self.cycle_position: int = 0
        self.cycle_count:    int = 0

    def observe(self, card_name: str, elapsed: float) -> None:
        card = card_name.lower().strip()
        if card not in self.enemy_deck:
            if len(self.enemy_deck) < self.FULL_DECK_SIZE:
                self.enemy_deck.append(card)
                self.play_count[card] = 0
        if card in self.enemy_deck:
            self.last_seen[card] = elapsed
            self.play_count[card] = self.play_count.get(card, 0) + 1
            deck_len = len(self.enemy_deck)
            if deck_len > 0:
                pos = self.enemy_deck.index(card)
                if pos == self.cycle_position % deck_len:
                    self.cycle_position = (self.cycle_position + 1) % deck_len
                    if self.cycle_position == 0:
                        self.cycle_count += 1

    # ── V25 required interface ─────────────────────────────────────────────

    def cards_until(self, card: str) -> int:
        """How many cards until this card comes back around."""
        card = card.lower().strip()
        if card not in self.enemy_deck:
            return self.FULL_DECK_SIZE
        deck_len = max(1, len(self.enemy_deck))
        idx = self.enemy_deck.index(card)
        cur = self.cycle_position % deck_len
        return (idx - cur) if idx >= cur else (deck_len - cur + idx)

    def predict_next_card(self) -> str:
        """Best guess for the next card the enemy will play."""
        if not self.enemy_deck:
            return "UNKNOWN"
        return self.enemy_deck[self.cycle_position % len(self.enemy_deck)]

    def is_out_of_cycle(self, card: str) -> bool:
        """True if the card is NOT in the enemy's current 4-card hand."""
        card = card.lower().strip()
        if card not in self.enemy_deck:
            return True
        deck_len = len(self.enemy_deck)
        hand_indices = [(self.cycle_position + i) % deck_len for i in range(self.HAND_SIZE)]
        return self.enemy_deck.index(card) not in hand_indices

    # ── Extended helpers ──────────────────────────────────────────────────

    def predict_next_n(self, n: int) -> List[str]:
        if not self.enemy_deck:
            return ["UNKNOWN"] * n
        deck_len = len(self.enemy_deck)
        return [self.enemy_deck[(self.cycle_position + i) % deck_len] for i in range(n)]

    def spell_available(self) -> bool:
        return any(c in self.SPELLS for c in self.predict_next_n(self.HAND_SIZE))

    def building_available(self) -> bool:
        return any(c in self.BUILDINGS for c in self.predict_next_n(self.HAND_SIZE))

    def get_cycle_pressure(self) -> float:
        """0→1: how close is the enemy to their next win condition."""
        if not self.enemy_deck:
            return 0.5
        deck_len = max(1, len(self.enemy_deck))
        for i, card in enumerate(self.predict_next_n(deck_len)):
            if card in self.WIN_CONDS:
                return 1.0 - (i / deck_len)
        return 0.0

    def deck_confidence(self) -> float:
        return min(1.0, len(self.enemy_deck) / self.FULL_DECK_SIZE)


# =============================================================================
# BAYESIAN ARCHETYPE IDENTIFIER
# =============================================================================

class BayesianArchetypeIdentifier:
    PRIORS: Dict[str, float] = {
        "HOG_2.6_CYCLE": 0.18, "BEATDOWN_PUSH": 0.14, "LOG_BAIT": 0.12,
        "SIEGE": 0.08, "BRIDGE_SPAM": 0.10, "GRAVEYARD_CONTROL": 0.07,
        "LAVALOON": 0.10, "MINER_CONTROL": 0.09, "UNKNOWN": 0.12,
    }
    LIKELIHOODS: Dict[str, Dict[str, float]] = {
        "hog rider":    {"HOG_2.6_CYCLE": 0.95, "BRIDGE_SPAM": 0.30},
        "ram rider":    {"HOG_2.6_CYCLE": 0.70, "BRIDGE_SPAM": 0.50},
        "golem":        {"BEATDOWN_PUSH": 0.90},
        "giant":        {"BEATDOWN_PUSH": 0.60, "UNKNOWN": 0.30},
        "lava_hound":   {"LAVALOON": 0.90, "BEATDOWN_PUSH": 0.10},
        "balloon":      {"LAVALOON": 0.80, "BEATDOWN_PUSH": 0.20},
        "goblin barrel":{"LOG_BAIT": 0.90},
        "xbow":         {"SIEGE": 0.98},
        "mortar":       {"SIEGE": 0.90},
        "graveyard":    {"GRAVEYARD_CONTROL": 0.95},
        "miner":        {"MINER_CONTROL": 0.85, "LOG_BAIT": 0.20},
        "pekka":        {"BRIDGE_SPAM": 0.85},
        "sparky":       {"BRIDGE_SPAM": 0.30, "UNKNOWN": 0.70},
        "cannon":       {"HOG_2.6_CYCLE": 0.60, "SIEGE": 0.50},
        "tesla":        {"SIEGE": 0.70, "HOG_2.6_CYCLE": 0.30},
        "poison":       {"GRAVEYARD_CONTROL": 0.75, "MINER_CONTROL": 0.50},
        "ice_golem":    {"HOG_2.6_CYCLE": 0.70, "MINER_CONTROL": 0.40},
    }

    def __init__(self):
        self.posteriors: Dict[str, float] = dict(self.PRIORS)

    def update(self, card_name: str) -> None:
        card = card_name.lower().strip()
        lh = self.LIKELIHOODS.get(card, {})
        if not lh:
            return
        new = {arch: self.posteriors[arch] * lh.get(arch, 0.05)
               for arch in self.posteriors}
        total = sum(new.values())
        if total > 0:
            self.posteriors = {k: v / total for k, v in new.items()}

    def get_best(self) -> Tuple[str, float]:
        best = max(self.posteriors.items(), key=lambda x: x[1])
        return best[0], best[1]


# =============================================================================
# PREDICTION ENGINE
# =============================================================================

class PredictionEngine:
    """Predicts specific incoming threats using cycle + archetype + elixir + board."""

    def predict_hog_rider(self, tracker: "GameTracker") -> Tuple[bool, float]:
        c = tracker.rotation.cards_until("hog rider")
        if c <= 2 and tracker.real_enemy_elixir >= 4:
            confidence = 0.85 if c == 0 else 0.70
            return True, confidence
        if tracker.enemy_archetype in ("HOG_2.6_CYCLE",) and tracker.real_enemy_elixir >= 4:
            return True, 0.55
        return False, 0.0

    def predict_balloon(self, tracker: "GameTracker") -> Tuple[bool, float]:
        c = tracker.rotation.cards_until("balloon")
        if c <= 2 and tracker.real_enemy_elixir >= 5:
            return True, 0.75
        return False, 0.0

    def predict_goblin_barrel(self, tracker: "GameTracker") -> Tuple[bool, float]:
        c = tracker.rotation.cards_until("goblin barrel")
        if c == 0 and tracker.real_enemy_elixir >= 3:
            return True, 0.90
        if c <= 2 and tracker.real_enemy_elixir >= 3:
            return True, 0.70
        return False, 0.0

    def predict_graveyard(self, tracker: "GameTracker") -> Tuple[bool, float]:
        c = tracker.rotation.cards_until("graveyard")
        if c <= 1 and tracker.real_enemy_elixir >= 5:
            return True, 0.75
        return False, 0.0

    def predict_freeze(self, tracker: "GameTracker", enemies: List[Dict]) -> Tuple[bool, float]:
        c = tracker.rotation.cards_until("freeze")
        if c <= 1 and tracker.real_enemy_elixir >= 4:
            # Check if enemy has a push already on board
            enemy_in_our_half = any(e["center"][1] > 280 for e in enemies)
            if enemy_in_our_half:
                return True, 0.75
            return True, 0.50
        return False, 0.0

    def predict_golem_push(self, tracker: "GameTracker", enemies: List[Dict]) -> Tuple[bool, float]:
        golem_on_board = any("golem" in e.get("name", "").lower() for e in enemies)
        if golem_on_board:
            return True, 0.95
        c = tracker.rotation.cards_until("golem")
        if c <= 1 and tracker.real_enemy_elixir >= 8:
            return True, 0.80
        return False, 0.0

    def get_all_predictions(self, tracker: "GameTracker", enemies: List[Dict]) -> Dict[str, Tuple[bool, float]]:
        return {
            "hog_rider":     self.predict_hog_rider(tracker),
            "balloon":       self.predict_balloon(tracker),
            "goblin_barrel": self.predict_goblin_barrel(tracker),
            "graveyard":     self.predict_graveyard(tracker),
            "freeze":        self.predict_freeze(tracker, enemies),
            "golem_push":    self.predict_golem_push(tracker, enemies),
        }


# =============================================================================
# THREAT HEATMAPS
# =============================================================================

class ThreatHeatmap:
    """
    9×4 spatial heatmaps for tactical analysis.
    danger_map:   threat weight per cell from enemies
    pressure_map: troop density per cell
    spell_value_map: value of casting a spell in each cell
    """

    def __init__(self):
        self.danger_map    = np.zeros((9, 4), dtype=np.float32)
        self.pressure_map  = np.zeros((9, 4), dtype=np.float32)
        self.spell_value_map = np.zeros((9, 4), dtype=np.float32)

    def update(self, enemies: List[Dict], my_elixir: float, phase: str) -> None:
        self.danger_map.fill(0.0)
        self.pressure_map.fill(0.0)
        for e in enemies:
            x, y  = e["center"]
            col   = int(min(8, max(0, (x - 60) // 30)))
            row   = int(min(3, max(0, (y - 230) // 48)))
            name  = e.get("name", "").replace("_Card", "").lower()
            role  = CARD_PROFILES.get(name, {}).get("role", "CYCLE")
            threat = ROLE_THREAT_WEIGHTS.get(role, 2)
            self.danger_map[col, row]   += threat
            self.pressure_map[col, row] += 1.0
        # Spell value: high where danger is low and we have elixir
        self.spell_value_map = (self.danger_map + 0.1) * self.pressure_map * (my_elixir / 10.0)

    def best_defense_zone(self, side: str) -> Tuple[int, int]:
        col_range = range(0, 4) if side == "left" else range(5, 9)
        best_col, best_row, best_val = 4, 2, -1.0
        for c in col_range:
            for r in range(4):
                if self.danger_map[c, r] > best_val:
                    best_val = self.danger_map[c, r]
                    best_col, best_row = c, r
        return best_col, best_row

    def best_spell_zone(self) -> Tuple[int, int]:
        idx = int(np.argmax(self.spell_value_map))
        return divmod(idx, 4)

    def zone_to_coord(self, col: int, row: int) -> Tuple[int, int]:
        x = ZONE_XS[col] if col < len(ZONE_XS) else 200
        y = ZONE_YS[row] if row < len(ZONE_YS) else 300
        return (x, y)


# =============================================================================
# META KNOWLEDGE BASE
# =============================================================================

class MetaKnowledgeBase:
    """Maps observed cards to known meta decks and provides counter guidance."""

    META_DECKS: Dict[str, Dict] = {
        "hog_2.6": {
            "cards": ["hog rider", "ice_golem", "cannon", "musketeer",
                      "fireball", "zap", "ice_spirit", "skeletons"],
            "weakness": ["inferno_tower", "giant_snowball", "rocket"],
            "prevalence": 0.18,
        },
        "golem_beatdown": {
            "cards": ["golem", "night_witch", "lumberjack", "baby_dragon",
                      "lightning", "zap", "tornado", "elixir_collector"],
            "weakness": ["inferno_tower", "inferno_dragon", "pekka", "mini_pekka"],
            "prevalence": 0.12,
        },
        "xbow_siege": {
            "cards": ["xbow", "tesla", "archers", "log",
                      "fireball", "ice_spirit", "skeletons", "knight"],
            "weakness": ["miner", "goblin_barrel", "rocket", "balloon"],
            "prevalence": 0.09,
        },
        "lava_loon": {
            "cards": ["lava_hound", "balloon", "inferno_dragon", "mega_minion",
                      "tombstone", "zap", "arrows", "haste"],
            "weakness": ["inferno_tower", "electro_dragon", "musketeer"],
            "prevalence": 0.11,
        },
        "graveyard_poison": {
            "cards": ["graveyard", "poison", "knight", "archers",
                      "cannon_cart", "tornado", "log", "ice_spirit"],
            "weakness": ["tesla", "bomb_tower", "giant_snowball"],
            "prevalence": 0.08,
        },
        "pekka_bridge": {
            "cards": ["pekka", "electro_wizard", "battle_ram", "zap",
                      "poison", "dark_prince", "magic_archer", "mega_minion"],
            "weakness": ["inferno_tower", "inferno_dragon", "mini_pekka"],
            "prevalence": 0.10,
        },
    }

    META_TO_ARCHETYPE: Dict[str, str] = {
        "hog_2.6": "HOG_2.6_CYCLE", "golem_beatdown": "BEATDOWN_PUSH",
        "xbow_siege": "SIEGE", "lava_loon": "LAVALOON",
        "graveyard_poison": "GRAVEYARD_CONTROL", "pekka_bridge": "BRIDGE_SPAM",
    }

    def identify_deck(self, seen: List[str]) -> Tuple[str, float]:
        seen_set  = {c.lower() for c in seen}
        best_name = "UNKNOWN"
        best_conf = 0.0
        for dname, ddata in self.META_DECKS.items():
            deck_set = {c.lower() for c in ddata["cards"]}
            matches  = len(seen_set & deck_set)
            conf     = matches / max(1, len(deck_set)) * (1.0 + ddata.get("prevalence", 0))
            if conf > best_conf:
                best_conf = conf
                best_name = dname
        return best_name, float(np.clip(best_conf, 0.0, 1.0))

    def get_spell_discipline(self, archetype: str) -> Dict[str, List[str]]:
        disciplines: Dict[str, Dict[str, List[str]]] = {
            "hog_2.6": {
                "never_use_on": ["hog rider", "ram_rider"],
                "save_for": ["musketeer", "cluster_support"],
            },
            "golem_beatdown": {
                "never_use_on": ["small_troops"],
                "save_for": ["golem", "support_cluster"],
            },
            "xbow_siege": {
                "never_use_on": ["knight"],
                "save_for": ["xbow_building"],
            },
            "lava_loon": {
                "never_use_on": ["lava_hound"],
                "save_for": ["balloon_cluster"],
            },
        }
        return disciplines.get(archetype, {"never_use_on": [], "save_for": ["cluster"]})


# =============================================================================
# PSYCH WARFARE ENGINE
# =============================================================================

class PsychWarfareEngine:
    """Bait and patience tactics."""

    def __init__(self):
        self.bait_attempts   = 0
        self.bait_successes  = 0
        self.last_bait_time  = 0.0
        self.spell_timings: List[float] = []

    def should_bait_spell(self, elapsed: float, archetype: str) -> Tuple[bool, str, Tuple[int, int]]:
        if elapsed - self.last_bait_time < 8.0:
            return False, "", (0, 0)
        if archetype == "LOG_BAIT":
            self.last_bait_time = elapsed
            self.bait_attempts += 1
            return True, "goblin barrel", PRECISION_PLACEMENTS.get("barrel_corner_l", (68, 175))
        if archetype == "HOG_2.6_CYCLE":
            self.last_bait_time = elapsed
            self.bait_attempts += 1
            return True, "skeletons", (200, 320)
        return False, "", (0, 0)

    def record_spell(self, timing: float) -> None:
        self.spell_timings.append(timing)
        if len(self.spell_timings) > 10:
            self.spell_timings.pop(0)

    def should_wait(self, elixir: float, phase: str, game_phase: str,
                    enemy_elixir: float, elapsed: float) -> bool:
        if elixir < 4.5 and phase not in ("DEFEND", "SOFT_DEFEND", "FULL_RUSH"):
            return True
        if game_phase == "NORMAL" and (120 - elapsed) < 30 and elixir < 9.0:
            return True
        return False


# =============================================================================
# ADVANCED VISION SYSTEM
# =============================================================================

class AdvancedVisionSystem:
    """HP bars, elixir bars, rage detection."""

    def detect_tower_hp(self, screenshot: np.ndarray, position: str) -> float:
        """[FIX C] Multi-colour HP bar: green + yellow + red."""
        if not CV2_AVAILABLE or screenshot is None:
            return 1.0
        regions = {
            "enemy_left":  (85, 140, 145, 155),
            "enemy_right": (260, 140, 320, 155),
            "my_left":     (85, 435, 145, 450),
            "my_right":    (260, 435, 320, 450),
        }
        region = regions.get(position)
        if region is None:
            return 1.0
        x1, y1, x2, y2 = region
        try:
            h, w = screenshot.shape[:2]
            bar = screenshot[y1:min(h, y2), x1:min(w, x2)]
            if bar.size == 0:
                return 1.0
            r, g, b = bar[:, :, 0].astype(int), bar[:, :, 1].astype(int), bar[:, :, 2].astype(int)
            green  = (g - b > 30)
            yellow = (r > 150) & (g > 150) & (b < 100)
            red    = (r > 150) & (g < 100) & (b < 100)
            mask   = green | yellow | red
            return float(np.clip(np.sum(mask) / max(1, bar.shape[0] * bar.shape[1]), 0.0, 1.0))
        except Exception:
            return 1.0

    def read_elixir(self, screenshot: np.ndarray, region: Tuple[int, int, int, int],
                    fallback: float = 5.0) -> float:
        if not CV2_AVAILABLE or screenshot is None:
            return fallback
        y1, y2, x1, x2 = region
        try:
            bar = screenshot[y1:y2, x1:x2]
            if bar.size == 0:
                return fallback
            bar_width   = x2 - x1
            results: List[float] = []
            hsv = cv2.cvtColor(bar, cv2.COLOR_RGB2HSV)
            lo, hi = np.array([130, 40, 40], np.uint8), np.array([150, 255, 255], np.uint8)
            mask = cv2.inRange(hsv, lo, hi)
            cols = np.where(mask > 0)[1]
            if len(cols) > 0:
                results.append(float(np.max(cols)) / bar_width * 10.0)
            sat = hsv[:, :, 1]
            high = np.where(sat > 60)[1]
            if len(high) > 0:
                results.append(float(np.max(high)) / bar_width * 10.0)
            if not results:
                return fallback
            return float(np.clip(np.median(results), 0.0, 10.0))
        except Exception:
            return fallback

    def read_my_elixir(self, screenshot: np.ndarray) -> float:
        return self.read_elixir(screenshot, (580, 595, 10, 350), 7.0)

    def read_enemy_elixir(self, screenshot: np.ndarray) -> float:
        return self.read_elixir(screenshot, (5, 18, 10, 350), 5.0)

    def detect_rage(self, screenshot: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        if not CV2_AVAILABLE or screenshot is None:
            return False
        try:
            x1, y1, x2, y2 = bbox
            h, w = screenshot.shape[:2]
            region = screenshot[max(0, y1-10):min(h, y2+10), max(0, x1-10):min(w, x2+10)]
            if region.size == 0:
                return False
            orange = (region[:, :, 0].astype(int) > 200) & \
                     (region[:, :, 1].astype(int) < 120) & \
                     (region[:, :, 2].astype(int) < 80)
            return int(np.sum(orange)) > 15
        except Exception:
            return False


# =============================================================================
# TOWER DAMAGE TRACKER
# =============================================================================

class TowerDamageTracker:
    """Computes per-step delta reward from tower HP changes."""

    def __init__(self):
        self.prev_enemy_hp: Dict[str, float] = {"left": 1.0, "right": 1.0}
        self.prev_my_hp:    Dict[str, float] = {"left": 1.0, "right": 1.0}

    def compute_delta(self, cur_enemy: Dict[str, float], cur_my: Dict[str, float]) -> float:
        reward = 0.0
        for side in ("left", "right"):
            dmg_dealt    = self.prev_enemy_hp[side] - cur_enemy.get(side, 1.0)
            dmg_received = self.prev_my_hp[side]    - cur_my.get(side, 1.0)
            reward += dmg_dealt    * 30.0
            reward -= dmg_received * 20.0
            if cur_enemy.get(side, 1.0) <= 0.0 and self.prev_enemy_hp[side] > 0.0:
                reward += 80.0
            if cur_my.get(side, 1.0) <= 0.0 and self.prev_my_hp[side] > 0.0:
                reward -= 60.0
        self.prev_enemy_hp.update(cur_enemy)
        self.prev_my_hp.update(cur_my)
        return reward


# =============================================================================
# TACTICAL PLANNING
# =============================================================================

class TacticalObjective(Enum):
    DEFEND       = 1
    COUNTER_PUSH = 2
    PRESSURE     = 3
    SPELL_CYCLE  = 4
    STALL        = 5
    PUNISH       = 6
    ALL_IN       = 7
    FINISH_TOWER = 8


class ActionPlan:
    def __init__(self, steps: List[TacticalObjective]):
        self.steps       = steps
        self.step_index  = 0
        self.step_frames = 0

    def current(self) -> TacticalObjective:
        return self.steps[self.step_index]

    def tick(self, board_changed: bool) -> None:
        self.step_frames += 1
        should_advance = self.step_frames > 25 or board_changed
        if should_advance and self.step_index < len(self.steps) - 1:
            self.step_index  += 1
            self.step_frames  = 0

    def is_complete(self) -> bool:
        return self.step_index >= len(self.steps) - 1


class StrategicBrain:
    """
    Multi-step objective planner.
    Evaluates board state each tick and returns the active ActionPlan.
    """

    def __init__(self):
        self.current_plan: Optional[ActionPlan] = None
        self.last_obj: Optional[TacticalObjective] = None

    def update(self, tracker: "GameTracker", enemies: List[Dict],
               my_hp: float, enemy_hp: float) -> TacticalObjective:
        # Choose primary objective
        if my_hp < 0.25:
            obj = TacticalObjective.DEFEND
        elif enemy_hp < 0.25:
            obj = TacticalObjective.FINISH_TOWER
        elif tracker.is_punish_window:
            obj = TacticalObjective.PUNISH
        elif tracker.game_phase in ("TRIPLE", "OVERTIME"):
            obj = TacticalObjective.ALL_IN
        elif tracker.current_phase == "SPELL_CYCLE":
            obj = TacticalObjective.SPELL_CYCLE
        elif tracker.counter_push_active:
            obj = TacticalObjective.COUNTER_PUSH
        elif len(enemies) >= 2 and any(e["center"][1] > 310 for e in enemies):
            obj = TacticalObjective.DEFEND
        elif tracker.real_my_elixir > 7.5:
            obj = TacticalObjective.PRESSURE
        else:
            obj = TacticalObjective.STALL

        board_changed = (obj != self.last_obj)
        self.last_obj = obj

        if self.current_plan is None or obj != self.current_plan.steps[0]:
            if obj == TacticalObjective.DEFEND:
                self.current_plan = ActionPlan([TacticalObjective.DEFEND, TacticalObjective.COUNTER_PUSH,
                                                TacticalObjective.PRESSURE])
            elif obj == TacticalObjective.PUNISH:
                self.current_plan = ActionPlan([TacticalObjective.PUNISH, TacticalObjective.PRESSURE,
                                                TacticalObjective.FINISH_TOWER])
            elif obj == TacticalObjective.COUNTER_PUSH:
                self.current_plan = ActionPlan([TacticalObjective.COUNTER_PUSH, TacticalObjective.PRESSURE])
            else:
                self.current_plan = ActionPlan([obj])

        self.current_plan.tick(board_changed)
        return self.current_plan.current()


# =============================================================================
# GAME TRACKER  (unified, thread-safe)
# =============================================================================

class GameTracker:
    """Central state accumulator shared across all systems."""

    def __init__(self):
        self.start_time: Optional[float] = None
        # Elixir
        self.real_my_elixir    = 7.0
        self.real_enemy_elixir = 5.0
        self.my_estimated_elixir = 7.0
        self.total_my_spent    = 0.0
        self.total_enemy_est   = 0.0
        self.elixir_advantage  = 0.0
        # Phase
        self.current_phase = "STALL"
        self.game_phase    = "NORMAL"
        self.side          = "left"
        # Push state
        self.counter_push_active = False
        self.counter_push_side   = "left"
        self.last_defense_time   = 0.0
        # Tower HP
        self.tower_hp_my_left    = 1.0
        self.tower_hp_my_right   = 1.0
        self.tower_hp_enemy_left = 1.0
        self.tower_hp_enemy_right= 1.0
        # History
        self.enemy_history: collections.deque = collections.deque(maxlen=20)
        self.enemy_archetype = "UNKNOWN"
        self.last_enemy_seen_time = 0.0
        # Subsystems
        self.rotation      = EnemyRotationTracker()
        self.bayesian      = BayesianArchetypeIdentifier()
        self.heatmap       = ThreatHeatmap()
        self.prediction    = PredictionEngine()
        self.strategic     = StrategicBrain()
        self.meta_kb       = MetaKnowledgeBase()
        self.psych         = PsychWarfareEngine()
        self.vision        = AdvancedVisionSystem()
        self.tower_damage  = TowerDamageTracker()
        # Misc
        self.frame_counter       = 0
        self.cards_played_match  = 0
        self.king_activated      = False
        self.meta_confidence     = 0.0
        self._prev_yolo: list    = []
        self.last_seen_time: Dict[str, float] = {}

    @property
    def is_punish_window(self) -> bool:
        return self.real_enemy_elixir <= 2.5 and self.real_my_elixir >= 5.0

    def update(self, elapsed: float, enemies: List[Dict],
               yolo_data: List[Dict], img: Optional[np.ndarray]) -> None:
        self.frame_counter += 1

        # Game phase
        if elapsed < 30:
            self.game_phase = "OPENING"
        elif elapsed < 120:
            self.game_phase = "NORMAL"
        elif elapsed < 180:
            self.game_phase = "DOUBLE"
        elif elapsed < 210:
            self.game_phase = "TRIPLE"
        else:
            self.game_phase = "OVERTIME"

        rate = {"OPENING": 0.4, "NORMAL": 0.5, "DOUBLE": 1.0, "TRIPLE": 1.4, "OVERTIME": 1.8}.get(
            self.game_phase, 0.5)
        self.my_estimated_elixir = min(10.0, self.my_estimated_elixir + rate * 0.35)
        self.real_my_elixir      = min(10.0, self.real_my_elixir      + rate * 0.35)

        # Observe enemies
        now = time.time()
        for e in enemies:
            name = e["name"].replace("_Card", "").replace(".", "").lower()
            if not self.enemy_history or self.enemy_history[-1] != name:
                self.enemy_history.append(name)
                self.rotation.observe(name, elapsed)
                self.bayesian.update(name)
                cost = CARD_PROFILES.get(name, {}).get("cost", 3)
                self.total_enemy_est += cost
                self.last_seen_time[name] = now
                self.last_enemy_seen_time = elapsed

        # Archetype
        best, conf = self.bayesian.get_best()
        if conf > 0.45:
            self.enemy_archetype = best
        else:
            h = " ".join(self.enemy_history)
            if "hog" in h or "ram" in h:
                self.enemy_archetype = "HOG_2.6_CYCLE"
            elif "golem" in h or "giant" in h or "lava" in h:
                self.enemy_archetype = "BEATDOWN_PUSH"
            elif "goblin barrel" in h or "princess" in h:
                self.enemy_archetype = "LOG_BAIT"
            elif "xbow" in h or "mortar" in h:
                self.enemy_archetype = "SIEGE"
            else:
                self.enemy_archetype = "UNKNOWN"

        # Meta KB supplement
        deck_name, meta_conf = self.meta_kb.identify_deck(list(self.enemy_history))
        if meta_conf > 0.55:
            self.meta_confidence = meta_conf
            mapped = self.meta_kb.META_TO_ARCHETYPE.get(deck_name)
            if mapped:
                self.enemy_archetype = mapped

        # Side
        if enemies:
            lead = max(enemies, key=lambda e: ROLE_THREAT_WEIGHTS.get(
                CARD_PROFILES.get(e["name"].replace("_Card", "").lower(), {}).get("role", "CYCLE"), 2))
            self.side = "right" if lead["center"][0] < 180 else "left"

        # Elixir advantage
        self.elixir_advantage = float(np.clip(self.total_my_spent - self.total_enemy_est, -20.0, 20.0))

        # Tactical phase
        total_threat = sum(ROLE_THREAT_WEIGHTS.get(
            CARD_PROFILES.get(e["name"].replace("_Card", "").lower(), {}).get("role", "CYCLE"), 2)
            for e in enemies)

        if self.tower_hp_my_left < 0.20 or self.tower_hp_my_right < 0.20:
            self.current_phase = "FULL_DEFENSE"
        elif self.game_phase in ("TRIPLE", "OVERTIME"):
            self.current_phase = "FULL_RUSH"
        elif total_threat >= 8:
            self.current_phase = "DEFEND"
        elif total_threat >= 4:
            self.current_phase = "SOFT_DEFEND"
        elif self.is_punish_window or self.my_estimated_elixir >= 7.0:
            self.current_phase = "PRESSURE"
        elif self.my_estimated_elixir >= 4.5:
            self.current_phase = "BUILD"
        else:
            self.current_phase = "STALL"

        # Counter push
        if self.current_phase in ("DEFEND", "SOFT_DEFEND") and total_threat < 2 and \
                elapsed - self.last_defense_time > 3.0:
            if ARCHETYPE_COUNTERS.get(self.enemy_archetype, {}).get("counter_push", True):
                self.counter_push_active = True
                self.counter_push_side   = self.side
        if total_threat >= 3:
            self.counter_push_active   = False
            self.last_defense_time     = elapsed

        # Heatmap
        self.heatmap.update(enemies, self.real_my_elixir, self.current_phase)

        # Tower HP vision (every 3 frames)
        if self.frame_counter % 3 == 0 and img is not None:
            self.tower_hp_my_left     = self.vision.detect_tower_hp(img, "my_left")
            self.tower_hp_my_right    = self.vision.detect_tower_hp(img, "my_right")
            self.tower_hp_enemy_left  = self.vision.detect_tower_hp(img, "enemy_left")
            self.tower_hp_enemy_right = self.vision.detect_tower_hp(img, "enemy_right")
            # Elixir vision
            ocr_my    = self.vision.read_my_elixir(img)
            ocr_enemy = self.vision.read_enemy_elixir(img)
            if ocr_my >= 0:
                self.real_my_elixir      = ocr_my
                self.my_estimated_elixir = ocr_my
            if ocr_enemy >= 0:
                self.real_enemy_elixir = ocr_enemy

    def spend_elixir(self, amount: float) -> None:
        self.my_estimated_elixir = max(0.0, self.my_estimated_elixir - amount)
        self.real_my_elixir      = max(0.0, self.real_my_elixir      - amount)
        self.total_my_spent     += amount
        self.cards_played_match += 1


# =============================================================================
# STATE EXTRACTION  [FIX B] 250-dim with hand cards  [FIX D] thread-safe
# =============================================================================

_tracker_lock = threading.Lock()


def identify_card(yolo_data: List[Dict], idx: int) -> str:
    """Identify a hand card by index (left-to-right order in bottom bar)."""
    hand = sorted(
        [p for p in yolo_data if "_Card" in p.get("name", "") and p["center"][1] > 510],
        key=lambda x: x["center"][0]
    )
    if idx < len(hand):
        raw = hand[idx]["name"]
        mapped = YOLO_TO_BOT_NAMES.get(raw)
        if mapped:
            return mapped
        return raw.replace("_Card", "").replace(".", "").lower().replace("_", " ")
    return "UNKNOWN"


def extract_state(yolo_data: List[Dict], hand_names: List[str], tracker: GameTracker) -> np.ndarray:
    """Build 250-dim state vector. [FIX D] reads tracker under lock snapshot."""
    sv = np.zeros(250, dtype=np.float32)
    idx = 0

    # Entities [0:200] — 40 × 5 dims
    entities = [d for d in yolo_data if "_Card" not in d.get("name", "")]
    for ent in entities[:40]:
        if idx >= 200:
            break
        is_enemy = ent.get("team") == "enemy" or "Enemy" in ent["name"]
        name     = ent["name"].replace("_Card", "").lower()
        role     = CARD_PROFILES.get(name, {}).get("role", "UNKNOWN")
        sv[idx]   = 1.0 if is_enemy else -1.0
        sv[idx+1] = ent["center"][0] / 360.0
        sv[idx+2] = ent["center"][1] / 600.0
        sv[idx+3] = ROLE_TO_ID.get(role, 0.1)
        sv[idx+4] = 1.0 if CARD_PROFILES.get(name, {}).get("air", False) else 0.0
        idx += 5

    # [FIX D] snapshot under lock
    with _tracker_lock:
        _elixir    = tracker.real_my_elixir
        _side      = tracker.side
        _phase     = tracker.current_phase
        _gphase    = tracker.game_phase
        _elixadv   = tracker.elixir_advantage
        _archetype = tracker.enemy_archetype
        _hist      = list(tracker.enemy_history)[:8]
        _cycpress  = tracker.rotation.get_cycle_pressure()
        _post      = dict(tracker.bayesian.posteriors)
        _eelix     = tracker.real_enemy_elixir
        _countpush = tracker.counter_push_active

    sv[200] = _elixir / 10.0
    sv[201] = 1.0 if _side == "left" else 0.0
    sv[202] = 1.0 if _phase == "DEFEND"       else 0.0
    sv[203] = 1.0 if _phase == "SOFT_DEFEND"  else 0.0
    sv[204] = 1.0 if _phase in ("PRESSURE", "BUILD") else 0.0
    sv[205] = 1.0 if _phase == "SPELL_CYCLE"  else 0.0
    sv[206] = 1.0 if _countpush               else 0.0
    sv[207] = 1.0 if _phase == "FULL_RUSH"    else 0.0
    sv[208] = {"OPENING": 0.0, "NORMAL": 0.3, "DOUBLE": 0.7, "TRIPLE": 0.9, "OVERTIME": 1.0}.get(_gphase, 0.0)
    sv[209] = float(np.clip(_elixadv / 20.0, -1.0, 1.0))
    sv[210] = {
        "UNKNOWN": 0.0, "HOG_2.6_CYCLE": 0.2, "BEATDOWN_PUSH": 0.35,
        "LOG_BAIT": 0.50, "SIEGE": 0.60, "BRIDGE_SPAM": 0.72,
        "GRAVEYARD_CONTROL": 0.82, "LAVALOON": 0.90, "MINER_CONTROL": 1.0,
    }.get(_archetype, 0.0)
    for i, card in enumerate(_hist):
        sv[211 + i] = ROLE_TO_ID.get(CARD_PROFILES.get(card, {}).get("role", "UNKNOWN"), 0.1)
    sv[219] = _cycpress
    ARCH_ORDER = ["HOG_2.6_CYCLE", "BEATDOWN_PUSH", "LOG_BAIT", "SIEGE", "BRIDGE_SPAM",
                  "GRAVEYARD_CONTROL", "LAVALOON", "MINER_CONTROL", "UNKNOWN"]
    total_post = sum(_post.values()) or 1.0
    for i, arch in enumerate(ARCH_ORDER):
        sv[220 + i] = _post.get(arch, 0.0) / total_post
    sv[229] = float(np.clip(_eelix / 10.0, 0.0, 1.0))
    # [FIX B] Hand cards [230:250]
    for slot, name in enumerate(hand_names[:4]):
        base = 230 + slot * 5
        prof = CARD_PROFILES.get(name.lower(), {})
        sv[base]   = prof.get("cost", 3) / 10.0
        sv[base+1] = ROLE_TO_ID.get(prof.get("role", "UNKNOWN"), 0.1)
        sv[base+2] = MECHANIC_TO_ID.get(prof.get("mechanic", "UNKNOWN"), 0.1)
        sv[base+3] = 1.0 if prof.get("role") == "WIN_CONDITION" else 0.0
        sv[base+4] = 1.0 if "SPELL" in prof.get("role", "") else 0.0
    return sv


# =============================================================================
# PLACEMENT UTILITIES
# =============================================================================

def get_defense_placement(side: str, depth: int = 0) -> Tuple[int, int]:
    zones = DEFENSE_ZONES.get(side, DEFENSE_ZONES["center"])
    return zones[max(0, min(depth, len(zones) - 1))]


def get_attack_placement(side: str, zone_idx: int) -> Tuple[int, int]:
    col = min(zone_idx, 3) if side == "left" else (min(5 + zone_idx, 8) if side == "right" else 4 + zone_idx % 3)
    col = max(0, min(col, 8))
    zone_i = 2 * 9 + col
    return SPATIAL_ZONES_9[zone_i] if zone_i < len(SPATIAL_ZONES_9) else (200, 300)


def get_inferno_melt_placement(enemies: List[Dict], side: str) -> Tuple[int, int]:
    tanks = [e for e in enemies if CARD_PROFILES.get(
        e["name"].replace("_Card", "").lower(), {}).get("role") in ("TANK", "WIN_CONDITION")]
    if not tanks:
        return get_defense_placement(side, 1)
    biggest = max(tanks, key=lambda t: CARD_PROFILES.get(
        t["name"].replace("_Card", "").lower(), {}).get("cost", 0))
    tx, ty = biggest["center"]
    offset_x = -20 if tx > 180 else 20
    return (int(max(60, min(300, tx + offset_x))), int(min(450, ty + 30)))


def get_best_spell_coord(enemies: List[Dict], spell_name: str, target_type: str,
                          side: str) -> Tuple[int, int]:
    SPLASH = {"fireball": 40, "rocket": 50, "arrows": 60, "zap": 35,
              "poison": 55, "freeze": 50, "log": 25}
    r = SPLASH.get(spell_name, 40)
    if target_type == "tower_snipe":
        return ENEMY_TOWER_L_RAW if side == "left" else ENEMY_TOWER_R_RAW
    if not enemies:
        return ENEMY_TOWER_L_RAW if side == "left" else ENEMY_TOWER_R_RAW
    if target_type == "tank":
        tanks = [e for e in enemies if CARD_PROFILES.get(
            e["name"].lower(), {}).get("role") in ("TANK", "WIN_CONDITION")]
        if tanks:
            return tanks[0]["center"]
    if target_type == "swarm":
        swarms = [e for e in enemies if CARD_PROFILES.get(
            e["name"].lower(), {}).get("role") in ("CYCLE", "AIR_DEFENSE")]
        if len(swarms) >= 2:
            return (int(np.mean([e["center"][0] for e in swarms])),
                    int(np.mean([e["center"][1] for e in swarms])))
    if len(enemies) >= 2:
        best, best_hits = enemies[0]["center"], 0
        for cx, cy in [e["center"] for e in enemies[:8]]:
            hits = sum(1 for e in enemies if abs(e["center"][0]-cx) <= r and abs(e["center"][1]-cy) <= r)
            if hits > best_hits:
                best_hits, best = hits, (cx, cy)
        return best
    xs = [e["center"][0] for e in enemies[:5]]
    ys = [e["center"][1] for e in enemies[:5]]
    return (int(np.mean(xs)), int(np.mean(ys)))


def should_activate_king(enemies: List[Dict], available_names: List[str],
                          king_activated: bool) -> bool:
    if king_activated:
        return False
    activation_cards = {"cannon", "tesla", "ice_spirit", "tornado"}
    dangerous = any("hog" in e["name"].lower() or "ram" in e["name"].lower() for e in enemies)
    has_activator = any(c in activation_cards for c in available_names)
    if "tornado" in available_names and enemies:
        side_enemies = [e for e in enemies if e["center"][0] < 80 or e["center"][0] > 280]
        if side_enemies:
            return True
    return dangerous and has_activator


def check_split_push(yolo_data: List[Dict], tracker: GameTracker,
                     available_names: List[str]) -> Optional[Tuple[str, Tuple[int, int]]]:
    main_side = tracker.side
    enemies = [d for d in yolo_data if (d.get("team") == "enemy" or "Enemy" in d["name"])
               and "_Card" not in d.get("name", "")]
    if len(enemies) < 2:
        return None
    main_x = 90 if main_side == "left" else 270
    on_main = sum(1 for e in enemies if abs(e["center"][0] - main_x) < 100)
    if on_main < len(enemies) * 0.7:
        return None
    cheap = [c for c in available_names
             if CARD_PROFILES.get(c, {}).get("cost", 99) <= 3
             and CARD_PROFILES.get(c, {}).get("role") in ("WIN_CONDITION", "CYCLE", "MINI_TANK")]
    if not cheap:
        return None
    opposite = "right" if main_side == "left" else "left"
    return (cheap[0], BRIDGE_LEFT_ATTACK if opposite == "left" else BRIDGE_RIGHT_ATTACK)


def check_freeze_combo(available_names: List[str], yolo_data: List[Dict]) -> bool:
    if "freeze" not in available_names:
        return False
    win_conds = {"balloon", "hog rider", "hog", "giant", "golem"}
    has_wc = any(c in win_conds for c in available_names)
    if not has_wc:
        return False
    friendly_pushing = any(
        not (d.get("team") == "enemy" or "Enemy" in d["name"])
        and d["center"][1] < 280 and "_Card" not in d.get("name", "")
        for d in yolo_data
    )
    return friendly_pushing


# =============================================================================
# PLACEMENT ENGINE  (V24 Objective-Driven + V23 special-case overrides)
# =============================================================================

def get_placement(
    tracker: GameTracker,
    objective: TacticalObjective,
    card_name: str,
    profile: Dict,
    is_spell: bool,
    enemies: List[Dict],
    available_indices: List[int],
    spatial_zone: int,
    hand_names: List[str],
    yolo_data: List[Dict],
    img: Optional[np.ndarray],
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    """
    Returns (raw_coord, override_card_index) or (None, None).
    Priority:
      1. King tower activation
      2. Bait play (psych engine)
      3. Freeze+Push combo
      4. Inferno melt vs beatdown
      5. Precision cannon vs Hog
      6. Overtime all-in
      7. Split push pressure
      8. Objective-driven placement
    """
    raw: Optional[Tuple[int, int]] = None
    override_idx: Optional[int] = None

    # ── 1. King Tower Activation ─────────────────────────────────────────────
    if not tracker.king_activated and should_activate_king(enemies, hand_names, False):
        tracker.king_activated = True
        key = f"king_activation_{tracker.side}"
        raw = PRECISION_PLACEMENTS.get(key)
        if raw:
            return raw, None

    # ── 2. Bait play ──────────────────────────────────────────────────────────
    should_bait, bait_card, bait_pos = tracker.psych.should_bait_spell(
        tracker.last_enemy_seen_time, tracker.enemy_archetype)
    if should_bait and bait_card in hand_names:
        bait_idx = hand_names.index(bait_card) if bait_card in hand_names else None
        if bait_idx is not None and bait_idx in available_indices:
            return bait_pos, bait_idx

    # ── 3. Freeze + Push combo ────────────────────────────────────────────────
    if check_freeze_combo(hand_names, yolo_data):
        if card_name == "freeze":
            raw = ENEMY_TOWER_L_RAW if tracker.side == "left" else ENEMY_TOWER_R_RAW
            return raw, None
        if card_name in ("balloon", "hog rider", "hog", "giant", "golem"):
            raw = BRIDGE_LEFT_ATTACK if tracker.side == "left" else BRIDGE_RIGHT_ATTACK
            return raw, None

    # ── 4. Inferno melt vs tank push ──────────────────────────────────────────
    if enemies and "inferno" in card_name:
        top = enemies[0]["name"].lower()
        if any(t in top for t in ("golem", "giant", "lava", "royal_giant")):
            return get_inferno_melt_placement(enemies, tracker.side), None

    # ── 5. Precision cannon vs Hog ────────────────────────────────────────────
    if enemies and profile.get("role") == "BUILDING":
        top = enemies[0]["name"].lower()
        if "hog" in top or "ram" in top:
            key = f"cannon_vs_hog_{tracker.side}"
            prec = PRECISION_PLACEMENTS.get(key)
            if prec:
                return prec, None

    # ── 6. Overtime all-in ───────────────────────────────────────────────────
    if tracker.game_phase == "OVERTIME":
        raw = BRIDGE_LEFT_ATTACK if tracker.side == "left" else BRIDGE_RIGHT_ATTACK
        return raw, None

    # ── 7. Split push during pressure ────────────────────────────────────────
    if objective in (TacticalObjective.PRESSURE, TacticalObjective.COUNTER_PUSH):
        split = check_split_push(yolo_data, tracker, hand_names)
        if split:
            sc, scoord = split
            si = hand_names.index(sc) if sc in hand_names else None
            if si is not None and si in available_indices:
                return scoord, si

    # ── 8. Objective-driven placement ────────────────────────────────────────

    if objective == TacticalObjective.DEFEND or objective == TacticalObjective.COUNTER_PUSH and enemies:
        if enemies and any(e["center"][1] > 280 for e in enemies):
            closest = min(enemies, key=lambda e: abs(e["center"][1] - 400))
            cy = min(480, closest["center"][1] + 40)
            coord = (closest["center"][0], cy)
            for idx in available_indices:
                role = CARD_PROFILES.get(hand_names[idx] if idx < len(hand_names) else "", {}).get("role", "")
                if role in ("BUILDING", "TANK_KILLER", "AIR_DEFENSE"):
                    return coord, idx
            if available_indices:
                return coord, None
            return coord, None

    if objective in (TacticalObjective.PUNISH, TacticalObjective.PRESSURE, TacticalObjective.ALL_IN):
        coord = BRIDGE_LEFT_ATTACK if tracker.side == "left" else BRIDGE_RIGHT_ATTACK
        for idx in available_indices:
            n = hand_names[idx] if idx < len(hand_names) else ""
            p = CARD_PROFILES.get(n, {})
            if p.get("role") == "WIN_CONDITION" or p.get("cost", 5) <= 2:
                return coord, idx
        return coord, None

    if objective == TacticalObjective.SPELL_CYCLE and is_spell:
        archetype_meta = ARCHETYPE_COUNTERS.get(tracker.enemy_archetype, ARCHETYPE_COUNTERS["UNKNOWN"])
        spell_target   = archetype_meta.get("spell_target", "cluster")
        if tracker.tower_hp_enemy_left < 0.25 or tracker.tower_hp_enemy_right < 0.25:
            coord = ENEMY_TOWER_L_RAW if tracker.side == "left" else ENEMY_TOWER_R_RAW
        elif enemies:
            coord = get_best_spell_coord(enemies, card_name, spell_target, tracker.side)
        else:
            coord = ENEMY_TOWER_L_RAW if tracker.side == "left" else ENEMY_TOWER_R_RAW
        return coord, None

    if objective == TacticalObjective.FINISH_TOWER:
        coord = ENEMY_TOWER_L_RAW if tracker.side == "left" else ENEMY_TOWER_R_RAW
        for idx in available_indices:
            n = hand_names[idx] if idx < len(hand_names) else ""
            p = CARD_PROFILES.get(n, {})
            if p.get("role") == "WIN_CONDITION" or "SPELL" in p.get("role", ""):
                return coord, idx
        return coord, None

    if objective == TacticalObjective.COUNTER_PUSH:
        coord = BRIDGE_LEFT_ATTACK if tracker.counter_push_side == "left" else BRIDGE_RIGHT_ATTACK
        return coord, None

    if objective == TacticalObjective.STALL:
        if profile.get("cost", 5) <= 2:
            zone_coord = SPATIAL_ZONES_9[spatial_zone] if spatial_zone < len(SPATIAL_ZONES_9) else (200, 300)
            return zone_coord, None
        return None, None

    # Fallback: spatial zone from DQN
    coord = SPATIAL_ZONES_9[spatial_zone] if spatial_zone < len(SPATIAL_ZONES_9) else (200, 300)
    return coord, None


# =============================================================================
# REWARD SYSTEM
# =============================================================================

def compute_reward(tracker: GameTracker, profile: Dict, phase: str,
                   is_counter: bool, enemies_before: List[Dict], enemies_after: List[Dict],
                   tower_delta: float, cycle_pressure: float,
                   predicted_hit: bool) -> float:
    reward = 0.0
    cost   = profile.get("cost", 3)

    # Tower HP delta (from TowerDamageTracker)
    reward += tower_delta

    # Elixir trade value
    cost_before = sum(CARD_PROFILES.get(e.get("name", "").replace("_Card", "").lower(), {}).get("cost", 0)
                      for e in enemies_before)
    cost_after  = sum(CARD_PROFILES.get(e.get("name", "").replace("_Card", "").lower(), {}).get("cost", 0)
                      for e in enemies_after)
    trade = (cost_before - cost_after) - cost
    reward += float(np.clip(trade * 1.5, -5.0, 10.0))

    # Defense value
    if phase in ("DEFEND", "SOFT_DEFEND", "FULL_DEFENSE"):
        if enemies_before and any(e["center"][1] > 350 for e in enemies_before):
            reward += 8.0

    # Counter card
    if is_counter:
        reward += 4.0

    # Counter push
    if tracker.counter_push_active and profile.get("role") in ("WIN_CONDITION", "TANK"):
        reward += 6.0

    # Cycle pressure prediction
    if cycle_pressure > 0.7 and is_counter:
        reward += 4.0

    # Prediction success
    if predicted_hit:
        reward += 12.0

    # Spell value
    if "SPELL" in profile.get("role", "") and enemies_before:
        reward += len(enemies_before) * 0.5

    # Penalties
    if phase == "STALL" and cost >= 5:
        reward -= 3.0
    if phase == "DEFEND" and not is_counter:
        if enemies_before and any(e["center"][1] > 380 for e in enemies_before):
            reward -= 2.0
    if profile.get("role") == "WIN_CONDITION" and tracker.real_enemy_elixir > 7.0:
        reward -= 4.0

    return float(np.clip(reward, -20.0, 20.0))


# =============================================================================
# MATCH ANALYTICS
# =============================================================================

class MatchAnalytics:
    """Tracks full per-match statistics for the report and dashboard."""

    def __init__(self):
        self.total_matches = 0
        self.total_wins    = 0
        self.rewards_hist: collections.deque  = collections.deque(maxlen=100)
        self.wins_hist:    collections.deque  = collections.deque(maxlen=100)
        self.dmg_dealt_hist:   collections.deque = collections.deque(maxlen=100)
        self.dmg_recv_hist:    collections.deque = collections.deque(maxlen=100)
        self.pos_trades_hist:  collections.deque = collections.deque(maxlen=100)
        self.neg_trades_hist:  collections.deque = collections.deque(maxlen=100)
        self.pred_hit_hist:    collections.deque = collections.deque(maxlen=100)
        self.pred_fail_hist:   collections.deque = collections.deque(maxlen=100)
        self.cp_rate_hist:     collections.deque = collections.deque(maxlen=100)
        self.def_rate_hist:    collections.deque = collections.deque(maxlen=100)
        self.placement_hist:   collections.deque = collections.deque(maxlen=100)
        self._cur: Dict[str, float] = {}

    def start_match(self) -> None:
        self._cur = {
            "reward": 0.0, "pos_trades": 0, "neg_trades": 0,
            "pred_hit": 0, "pred_fail": 0,
            "cp_ok": 0, "cp_total": 0,
            "def_ok": 0, "def_total": 0,
            "place_ok": 0, "place_total": 0,
        }

    def record_step(self, reward: float, trade: float, is_cp: bool, is_defend: bool,
                    pred_hit: bool, good_placement: bool) -> None:
        self._cur["reward"] += reward
        if trade > 0:
            self._cur["pos_trades"] += 1
        elif trade < 0:
            self._cur["neg_trades"] += 1
        if is_cp:
            self._cur["cp_total"]  += 1
            if reward > 0:
                self._cur["cp_ok"] += 1
        if is_defend:
            self._cur["def_total"]  += 1
            if reward > 0:
                self._cur["def_ok"] += 1
        if pred_hit:
            self._cur["pred_hit"] += 1
        else:
            self._cur["pred_fail"] += 1
        self._cur["place_total"] += 1
        if good_placement:
            self._cur["place_ok"] += 1

    def finish_match(self, won: bool, dmg_dealt: float, dmg_recv: float,
                     loss_val: Optional[float]) -> str:
        self.total_matches += 1
        if won:
            self.total_wins += 1
        m = self._cur
        self.rewards_hist.append(m["reward"])
        self.wins_hist.append(1 if won else 0)
        self.dmg_dealt_hist.append(dmg_dealt)
        self.dmg_recv_hist.append(dmg_recv)
        self.pos_trades_hist.append(m["pos_trades"])
        self.neg_trades_hist.append(m["neg_trades"])
        self.pred_hit_hist.append(m["pred_hit"])
        self.pred_fail_hist.append(m["pred_fail"])
        cp_r = m["cp_ok"] / max(1, m["cp_total"])
        de_r = m["def_ok"] / max(1, m["def_total"])
        pl_r = m["place_ok"] / max(1, m["place_total"])
        self.cp_rate_hist.append(cp_r)
        self.def_rate_hist.append(de_r)
        self.placement_hist.append(pl_r)
        return self._build_report(won, dmg_dealt, dmg_recv, loss_val, cp_r, de_r, pl_r)

    def _build_report(self, won: bool, dmg_dealt: float, dmg_recv: float,
                      loss_val: Optional[float], cp_r: float, de_r: float, pl_r: float) -> str:
        total   = max(1, self.total_matches)
        win_r   = self.total_wins / total * 100
        avg_rew = float(np.mean(list(self.rewards_hist))) if self.rewards_hist else 0.0
        prev_r  = float(np.mean(list(self.rewards_hist)[:-1])) if len(self.rewards_hist) > 1 else avg_rew
        improv  = ((avg_rew - prev_r) / max(0.01, abs(prev_r)) * 100) if prev_r != 0 else 0.0
        skill   = int(1000 + win_r / 100 * 500 + avg_rew * 0.5)
        m       = self._cur

        lines = [
            "",
            "=" * 60,
            "MATCH REPORT",
            "=" * 60,
            f"Result:                    {'WIN' if won else 'LOSS'}",
            f"Reward Earned:             {m['reward']:+.1f}",
            f"Tower Damage Dealt:        {int(dmg_dealt)}",
            f"Tower Damage Received:     {int(dmg_recv)}",
            f"Positive Trades:           {int(m['pos_trades'])}",
            f"Negative Trades:           {int(m['neg_trades'])}",
            f"Successful Predictions:    {int(m['pred_hit'])}",
            f"Failed Predictions:        {int(m['pred_fail'])}",
            f"Counter Push Success Rate: {cp_r*100:.0f}%",
            f"Defense Success Rate:      {de_r*100:.0f}%",
            f"Placement Accuracy:        {pl_r*100:.0f}%",
            f"Estimated Skill Score:     {skill}",
            f"Model Confidence:          {max(0, 100 - int(epsilon * 100))}%",
            f"Current Win Rate:          {win_r:.1f}%",
            f"Matches Played:            {self.total_matches}",
            f"Replay Buffer Size:        {len(memory)}",
            f"Training Loss:             {loss_val:.4f}" if loss_val else "Training Loss:             N/A",
            f"Average Reward Last 100:   {avg_rew:.1f}",
            f"Improvement Last 100:      {improv:+.1f}%",
            "=" * 60,
            "",
        ]
        return "\n".join(lines)


# =============================================================================
# TERMINAL DASHBOARD
# =============================================================================

class TerminalDashboard:
    def __init__(self, analytics: MatchAnalytics):
        self.analytics = analytics

    def _bar(self, value: float, max_v: float = 100.0, width: int = 22) -> str:
        filled = int(min(width, max(0, (value / max(0.01, max_v)) * width)))
        return "█" * filled + "░" * (width - filled)

    def print(self, tracker: GameTracker, loss_val: Optional[float]) -> None:
        a     = self.analytics
        total = max(1, a.total_matches)
        wr    = a.total_wins / total * 100
        ar    = float(np.mean(list(a.rewards_hist))) if a.rewards_hist else 0.0
        dr    = float(np.mean(list(a.def_rate_hist))) * 100 if a.def_rate_hist else 0.0
        cr    = float(np.mean(list(a.cp_rate_hist)))  * 100 if a.cp_rate_hist  else 0.0
        pl    = float(np.mean(list(a.placement_hist))) * 100 if a.placement_hist else 0.0
        conf  = max(0.0, 100 - epsilon * 100)
        eff   = float(np.clip(50 + tracker.elixir_advantage * 2.5, 0, 100))

        # Build 20-char trend bars from history
        def trend_bar(hist: collections.deque, scale: float = 1.0) -> str:
            vals = [v * scale for v in list(hist)[-20:]]
            if not vals:
                return "░" * 20
            mx = max(max(vals), 0.01)
            return "".join("█" if v / mx > 0.5 else "░" for v in vals)

        wr_trend = trend_bar(a.wins_hist, 100)
        rw_trend = trend_bar(a.rewards_hist)
        lp_trend = trend_bar(a.placement_hist, 100)

        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║            DRAGON ULTIMATE V25 — LIVE DASHBOARD             ║")
        print(f"║  Matches: {a.total_matches:<6}  Wins: {a.total_wins:<6}  Losses: {a.total_matches - a.total_wins:<10}║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print(f"║  Win Rate:           [{self._bar(wr)}] {wr:>5.1f}%  ║")
        print(f"║  Avg Reward:         [{self._bar(max(0,ar),200)}] {ar:>6.1f}  ║")
        print(f"║  Defense Rate:       [{self._bar(dr)}] {dr:>5.1f}%  ║")
        print(f"║  Counter Push:       [{self._bar(cr)}] {cr:>5.1f}%  ║")
        print(f"║  Placement Acc:      [{self._bar(pl)}] {pl:>5.1f}%  ║")
        print(f"║  Model Confidence:   [{self._bar(conf)}] {conf:>5.1f}%  ║")
        print(f"║  Elixir Efficiency:  [{self._bar(eff)}] {tracker.elixir_advantage:>+5.1f}   ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print(f"║  Win Rate Trend:     {wr_trend}          ║")
        print(f"║  Reward Trend:       {rw_trend}          ║")
        print(f"║  Learning Progress:  {lp_trend}          ║")
        if loss_val is not None:
            print(f"║  Training Loss:      {loss_val:<.4f}                                  ║")
        print(f"║  Epsilon:            {epsilon:<.4f}  BufferSize: {len(memory):<8}           ║")
        print(f"║  Archetype:          {tracker.enemy_archetype:<25}           ║")
        print(f"║  Next Enemy Card:    {tracker.rotation.predict_next_card():<20}                ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")


# =============================================================================
# GLOBAL INSTANCES  (initialised once, reset between matches)
# =============================================================================

dqn_agent    = DQNCrashBrain(state_dim=STATE_DIM, action_dim=ACTION_DIM)
target_dqn   = DQNCrashBrain(state_dim=STATE_DIM, action_dim=ACTION_DIM)
target_dqn.load_state_dict(dqn_agent.state_dict())
target_dqn.eval()
memory       = PrioritizedReplayBuffer(capacity=50000)
nstep_buf    = NStepBuffer(n=5, gamma=0.99)
tracker      = GameTracker()
analytics    = MatchAnalytics()
dashboard    = TerminalDashboard(analytics)

epsilon            = 0.9
EPSILON_END        = 0.02
EPSILON_DECAY      = 0.9995
global_step        = 0
prev_state         = None
last_action        = -1
last_actions_deque = collections.deque(maxlen=4)

_total_matches = 0
_total_wins    = 0

# Load saved model
if os.path.exists(MODEL_FILE):
    try:
        saved = torch.load(MODEL_FILE, weights_only=True)
        fw = saved.get("feature_net.0.weight")
        if fw is not None and fw.shape[1] == STATE_DIM:
            dqn_agent.load_state_dict(saved)
            target_dqn.load_state_dict(dqn_agent.state_dict())
            print(f"[V25] Model loaded from {MODEL_FILE}.")
        else:
            print(f"[V25] State dim mismatch — fresh start.")
    except Exception as e:
        print(f"[V25] Fresh start: {e}")

if os.path.exists(EPSILON_FILE):
    try:
        with open(EPSILON_FILE) as f:
            epsilon = max(EPSILON_END, float(f.read().strip()))
        print(f"[V25] Resumed epsilon={epsilon:.4f}")
    except Exception:
        pass


# =============================================================================
# MAIN TURN FUNCTION
# =============================================================================

def play_hybrid_turn(emulator, logger: Logger, battle_strategy) -> bool:
    global last_action, last_actions_deque, epsilon, global_step, prev_state, nstep_buf

    frame_start = time.time()
    try:
        screenshot = emulator.screenshot()
        if screenshot is None:
            return True
        img = np.asarray(screenshot)
    except Exception:
        return True

    yolo_data = get_yolo_predictions(img)

    # Robust card availability poll
    available_indices = check_which_cards_are_available(emulator, False, True)
    for _ in range(5):
        if available_indices:
            break
        interruptible_sleep(0.3)
        try:
            img = np.asarray(emulator.screenshot())
            yolo_data = get_yolo_predictions(img)
        except Exception:
            pass
        available_indices = check_which_cards_are_available(emulator, False, True)

    if not available_indices:
        emulator.click(200, 300)
        interruptible_sleep(0.5)
        available_indices = check_which_cards_are_available(emulator, False, True)
        if not available_indices:
            return True

    arena   = [d for d in yolo_data if "_Card" not in d.get("name", "")]
    enemies = sorted(
        [d for d in arena if d.get("team") == "enemy" or "Enemy" in d["name"]],
        key=lambda e: ROLE_THREAT_WEIGHTS.get(
            CARD_PROFILES.get(e["name"].replace("_Card", "").lower(), {}).get("role", "CYCLE"), 2),
        reverse=True,
    )

    elapsed = battle_strategy.get_elapsed_time()
    with _tracker_lock:
        tracker.update(elapsed, enemies, yolo_data, img)

    # Bait detection
    opponent_spell = None
    cur_names = {d["name"].lower() for d in yolo_data}
    for ind in ("fireball_impact", "zap_impact", "arrow_impact", "rocket_impact"):
        if ind in cur_names:
            opponent_spell = ind.replace("_impact", "")
    if opponent_spell and tracker.psych.bait_attempts > 0:
        since_bait = elapsed - tracker.psych.last_bait_time
        if since_bait < 3.0:
            tracker.psych.bait_successes += 1
            if memory.buffer:
                last_exp = memory.buffer[-1]
                memory.buffer[-1] = (last_exp[0], last_exp[1], last_exp[2] + 6.0, last_exp[3], last_exp[4])
    if opponent_spell:
        tracker.psych.record_spell(elapsed)

    hand_names = [identify_card(yolo_data, i) for i in range(4)]
    current_state = extract_state(yolo_data, hand_names, tracker)

    # Cycle pressure for adaptive epsilon
    cycle_pressure  = tracker.rotation.get_cycle_pressure()
    next_card       = tracker.rotation.predict_next_card()
    next_is_wc      = CARD_PROFILES.get(next_card, {}).get("role") == "WIN_CONDITION"
    eff_epsilon     = max(0.01, epsilon * 0.1) if (cycle_pressure > 0.7 and next_is_wc) else epsilon

    # DQN decision
    with torch.no_grad():
        state_t = torch.FloatTensor(current_state).unsqueeze(0).to(dqn_agent.device)
        qvals   = dqn_agent.model(state_t).clone()
        # [FIX F] Mask unavailable actions
        for i in range(36):
            if (i // 9) not in available_indices:
                qvals[0][i] = -1e9

        if random.random() < eff_epsilon:
            valid = [i for i in range(36) if (i // 9) in available_indices] + [WAIT_ACTION_INDEX]
            action = random.choice(valid) if valid else WAIT_ACTION_INDEX
        else:
            action = int(torch.argmax(qvals[0]).item())

        # Anti-loop
        recent_same = sum(1 for a in last_actions_deque if a == action)
        if recent_same >= 2:
            alt = sorted([i for i in range(36) if (i // 9) in available_indices and i != action],
                         key=lambda i: qvals[0][i].item(), reverse=True)
            if alt:
                action = alt[0]

    last_actions_deque.append(action)
    last_action = action

    # WAIT
    if action == WAIT_ACTION_INDEX:
        if prev_state is not None:
            nstep_buf.push(prev_state, WAIT_ACTION_INDEX, -0.05, current_state, False)
            ns = nstep_buf.get_nstep()
            if ns:
                memory.push(*ns)
        prev_state = current_state
        return True

    best_card_idx   = action // 9
    spatial_zone    = action % 9
    card_name       = identify_card(yolo_data, best_card_idx)
    profile         = CARD_PROFILES.get(card_name, {"role": "CYCLE", "cost": 2, "mechanic": "UNKNOWN"})
    is_spell        = "SPELL" in profile.get("role", "")

    # Tactical objective
    my_hp    = (tracker.tower_hp_my_left    + tracker.tower_hp_my_right)    / 2.0
    enemy_hp = (tracker.tower_hp_enemy_left + tracker.tower_hp_enemy_right) / 2.0
    objective = tracker.strategic.update(tracker, enemies, my_hp, enemy_hp)

    # Spell discipline check (skip if crisis)
    hp_critical = (tracker.tower_hp_my_left < 0.15 or tracker.tower_hp_my_right < 0.15)
    if is_spell and not hp_critical:
        disc = tracker.meta_kb.get_spell_discipline(tracker.enemy_archetype)
        top_enemy = enemies[0]["name"].lower().replace("_card", "") if enemies else ""
        if any(t in top_enemy for t in disc.get("never_use_on", [])):
            return True
        save_for = disc.get("save_for", [])
        arena_lower = [e["name"].lower() for e in enemies]
        save_present = any(any(s in n for n in arena_lower) for s in save_for)
        if save_for and not save_present and tracker.real_my_elixir < 9.0:
            return True

    # Placement
    is_counter = card_name in ARCHETYPE_COUNTERS.get(tracker.enemy_archetype, {}).get("best_counters", [])
    placement  = get_placement(tracker, objective, card_name, profile, is_spell,
                               enemies, available_indices, spatial_zone, hand_names, yolo_data, img)
    if placement is None:
        return True
    raw_coord, override_idx = placement
    if raw_coord is None:
        return True
    if override_idx is not None:
        best_card_idx = override_idx
        card_name     = identify_card(yolo_data, best_card_idx)
        profile       = CARD_PROFILES.get(card_name, profile)

    play_coord = map_coordinates(raw_coord[0], raw_coord[1])

    # Snapshot HP before play
    hp_before_enemy = {"left": tracker.tower_hp_enemy_left, "right": tracker.tower_hp_enemy_right}
    hp_before_my    = {"left": tracker.tower_hp_my_left,    "right": tracker.tower_hp_my_right}
    enemies_before  = list(enemies)

    # Execute
    card_coord = HAND_CARDS_COORDS[min(best_card_idx, 3)]
    emulator.click(card_coord[0], card_coord[1])
    interruptible_sleep(0.04)
    emulator.click(play_coord[0], play_coord[1])
    tracker.spend_elixir(profile.get("cost", 3))

    # Read results
    interruptible_sleep(0.07)
    try:
        img_after = np.asarray(emulator.screenshot())
        hp_after_enemy = {
            "left":  tracker.vision.detect_tower_hp(img_after, "enemy_left"),
            "right": tracker.vision.detect_tower_hp(img_after, "enemy_right"),
        }
        hp_after_my = {
            "left":  tracker.vision.detect_tower_hp(img_after, "my_left"),
            "right": tracker.vision.detect_tower_hp(img_after, "my_right"),
        }
        yolo_after = get_yolo_predictions(img_after)
        enemies_after = [d for d in yolo_after
                         if "_Card" not in d.get("name", "")
                         and (d.get("team") == "enemy" or "Enemy" in d["name"])]
    except Exception:
        hp_after_enemy = hp_before_enemy
        hp_after_my    = hp_before_my
        enemies_after  = []

    tower_delta = tracker.tower_damage.compute_delta(hp_after_enemy, hp_after_my)

    # Predictions
    all_preds   = tracker.prediction.get_all_predictions(tracker, enemies)
    pred_hog, _ = all_preds["hog_rider"]
    reward      = compute_reward(tracker, profile, tracker.current_phase, is_counter,
                                 enemies_before, enemies_after, tower_delta,
                                 cycle_pressure, pred_hog)

    # Trade value for analytics
    cost_before = sum(CARD_PROFILES.get(e.get("name", "").replace("_Card","").lower(), {}).get("cost", 0)
                      for e in enemies_before)
    cost_after2 = sum(CARD_PROFILES.get(e.get("name", "").replace("_Card","").lower(), {}).get("cost", 0)
                      for e in enemies_after)
    trade = (cost_before - cost_after2) - profile.get("cost", 3)

    analytics.record_step(
        reward=reward,
        trade=trade,
        is_cp=objective == TacticalObjective.COUNTER_PUSH,
        is_defend=objective == TacticalObjective.DEFEND,
        pred_hit=pred_hog and is_counter,
        good_placement=reward > 0,
    )

    # N-step push
    if prev_state is not None:
        nstep_buf.push(prev_state, action, reward, current_state, False)
        ns = nstep_buf.get_nstep()
        if ns:
            memory.push(*ns)
    prev_state = current_state

    # Online training
    global_step += 1
    last_loss: Optional[float] = None
    if global_step % ONLINE_TRAIN_FREQ == 0 and len(memory) >= ONLINE_BATCH_SIZE:
        try:
            last_loss = optimize_dqn_agent(dqn_agent, memory, ONLINE_BATCH_SIZE, target_net=target_dqn)
        except Exception as e:
            logger.change_status(f"[V25 TRAIN] {e}")

    # [FIX 6] Epsilon decays only on real card actions
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # [FIX 1] Target sync
    if global_step % TARGET_UPDATE_FREQ == 0:
        target_dqn.load_state_dict(dqn_agent.state_dict())

    # [FIX 7] Periodic checkpoint
    if global_step % 200 == 0:
        try:
            torch.save(dqn_agent.state_dict(), MODEL_FILE)
        except Exception:
            pass

    if global_step % 50 == 0:
        try:
            with open(EPSILON_FILE, "w") as f:
                f.write(str(epsilon))
        except Exception:
            pass

    frame_ms = (time.time() - frame_start) * 1000
    logger.change_status(
        f"[V25] {tracker.current_phase} | card={card_name} | "
        f"obj={objective.name} | ε={epsilon:.3f} | "
        f"elix={tracker.real_enemy_elixir:.1f} | "
        f"next={tracker.rotation.predict_next_card()} | "
        f"rew={reward:+.1f} | {frame_ms:.0f}ms"
    )
    return True


# =============================================================================
# FIGHT LOOP
# =============================================================================

class BattleStrategy:
    def __init__(self):
        self.start_time: Optional[float] = None

    def start_battle(self) -> None:
        self.start_time = time.time()
        global tracker, prev_state, nstep_buf
        with _tracker_lock:
            prev_state = None
            nstep_buf  = NStepBuffer(n=5, gamma=0.99)
            tracker    = GameTracker()
            tracker.start_time = self.start_time
        analytics.start_match()

    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0


def start_fight(emulator, logger, mode) -> bool:
    for _ in range(3):
        if check_if_on_clash_main_menu(emulator):
            emulator.click(203, 487)
            return True
        emulator.click(15, 300)
        interruptible_sleep(0.5)
    return True


def _fight_loop(emulator, logger: Logger, recording_flag: bool) -> bool:
    """[FIX E] Adaptive 120ms/frame target."""
    TARGET_FRAME = 0.120
    MIN_SLEEP    = 0.005
    battle       = BattleStrategy()
    battle.start_battle()
    while check_if_in_battle(emulator):
        t0 = time.time()
        try:
            play_hybrid_turn(emulator, logger, battle)
        except Exception as e:
            logger.change_status(f"[V25] Loop exception: {e}")
            interruptible_sleep(0.5)
        elapsed_f = time.time() - t0
        interruptible_sleep(max(MIN_SLEEP, TARGET_FRAME - elapsed_f))
    return True


def do_fight_state(emulator, logger, random_fight_mode, fight_mode_choosed,
                   called_from_launching=False, recording_flag=False) -> bool:
    if not wait_for_battle_start(emulator, logger):
        return False
    return _fight_loop(emulator, logger, recording_flag)


def do_2v2_fight_state(emulator, logger, random_fight_mode, recording_flag=False) -> bool:
    return do_fight_state(emulator, logger, random_fight_mode, "Classic 2v2", recording_flag=recording_flag)


def do_1v1_fight_state(emulator, logger, random_fight_mode, recording_flag=False) -> bool:
    return do_fight_state(emulator, logger, random_fight_mode, "Classic 1v1", recording_flag=recording_flag)


def do_trophy_road_fight_state(emulator, logger, random_fight_mode, recording_flag=False) -> bool:
    return do_fight_state(emulator, logger, random_fight_mode, "Trophy Road", recording_flag=recording_flag)


def find_post_battle_button(emulator):
    iar = emulator.screenshot()
    if iar is None:
        return None
    try:
        if pixel_is_equal(iar[550][200], [255, 255, 255], tol=30) or \
           pixel_is_equal(iar[545][178], [255, 187, 104], tol=30):
            return (200, 550, True)
    except Exception:
        pass
    coord = find_image(iar, "ok_post_battle_button", tolerance=0.85)
    if coord:
        return (coord[0], coord[1], True)
    coord = find_image(iar, "exit_battle_button", tolerance=0.9)
    if coord:
        return (coord[0], coord[1], False)
    return None


def end_fight_state(emulator, logger: Logger, recording_flag,
                    disable_win_tracker_toggle=True) -> bool:
    global prev_state, _total_matches, _total_wins

    logger.change_status("[V25] Match complete — running final backpropagation...")
    prev_state = None

    # Flush N-step transitions
    partials = nstep_buf.flush()
    for i, pt in enumerate(partials):
        s, a, g, ns, _ = pt
        is_last = (i == len(partials) - 1)
        memory.push(s, a, g, ns, is_last)
    nstep_buf.clear()

    timeout        = 60
    start_t        = time.time()
    last_click_t   = time.time()
    DEAD_ZONES     = [(15, 300), (385, 300), (200, 20)]
    model_updated  = False

    while time.time() - start_t < timeout:
        if check_if_on_clash_main_menu(emulator):
            logger.change_status("[V25] Returned to main menu. V25 ready.")
            return True

        res = find_post_battle_button(emulator)
        if res is not None:
            btn_x, btn_y, won_flag = res

            if not model_updated and memory.buffer:
                # [FIX 5] Graduated terminal reward
                if won_flag:
                    towers_destroyed = (
                        int(tracker.tower_hp_enemy_left  <= 0.0) +
                        int(tracker.tower_hp_enemy_right <= 0.0)
                    )
                    terminal_reward = 30.0 + towers_destroyed * 35.0
                else:
                    towers_lost = (
                        int(tracker.tower_hp_my_left  <= 0.0) +
                        int(tracker.tower_hp_my_right <= 0.0)
                    )
                    terminal_reward = -(20.0 + towers_lost * 15.0)

                last_exp = memory.buffer[-1]
                memory.buffer[-1] = (last_exp[0], last_exp[1],
                                     last_exp[2] + terminal_reward, last_exp[3], True)
                memory.priorities[-1] = memory.max_priority

                loss_val = optimize_dqn_agent(dqn_agent, memory, batch_size=64, target_net=target_dqn)

                if loss_val is not None:
                    # [FIX 1] Do NOT sync target network here — only every 1000 steps
                    torch.save(dqn_agent.state_dict(), MODEL_FILE)
                    with open(EPSILON_FILE, "w") as f:
                        f.write(str(epsilon))

                    # [FIX 9] CSV win-rate log
                    _total_matches += 1
                    if won_flag:
                        _total_wins += 1
                    win_rate = _total_wins / max(1, _total_matches)
                    try:
                        write_header = not os.path.exists(MATCH_STATS_FILE)
                        with open(MATCH_STATS_FILE, "a") as sf:
                            if write_header:
                                sf.write("match,result,terminal_reward,loss,epsilon,steps,win_rate,archetype\n")
                            sf.write(
                                f"{_total_matches},"
                                f"{'WIN' if won_flag else 'LOSS'},"
                                f"{terminal_reward:.1f},"
                                f"{loss_val:.4f},"
                                f"{epsilon:.4f},"
                                f"{global_step},"
                                f"{win_rate:.3f},"
                                f"{tracker.enemy_archetype}\n"
                            )
                    except Exception:
                        pass

                    # Tower damage totals for analytics
                    dmg_dealt = (
                        (1.0 - tracker.tower_hp_enemy_left)  * 3000 +
                        (1.0 - tracker.tower_hp_enemy_right) * 3000
                    )
                    dmg_recv  = (
                        (1.0 - tracker.tower_hp_my_left)  * 3000 +
                        (1.0 - tracker.tower_hp_my_right) * 3000
                    )
                    report = analytics.finish_match(won_flag, dmg_dealt, dmg_recv, loss_val)
                    print(report)
                    dashboard.print(tracker, loss_val)

                model_updated = True

            emulator.click(btn_x, btn_y)
            last_click_t = time.time()
            interruptible_sleep(0.5)
            continue

        if time.time() - last_click_t > 1.5:
            for dz in DEAD_ZONES:
                emulator.click(dz[0], dz[1])
                interruptible_sleep(0.1)
            emulator.click(CLASH_MAIN_DEADSPACE_COORD[0], CLASH_MAIN_DEADSPACE_COORD[1])
            last_click_t = time.time()

        interruptible_sleep(0.2)

    logger.change_status("[V25] Timeout in end_fight_state.")
    return False


# =============================================================================
# END OF FILE — DRAGON ULTIMATE V25
# =============================================================================