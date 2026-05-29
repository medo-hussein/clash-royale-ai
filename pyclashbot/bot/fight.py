"""
GRANDMASTER UNIVERSAL AI - VERSION 18.0 (HIERARCHICAL DRAGON AI)
Architecture: Strategy Layer (Phases/Trades) -> Tactical RL Layer (DQN) -> Execution Layer.
Features: Phase Management, Dynamic Trade Evaluation, Punish Windows, Repetition Penalty.
"""

import collections
import random
import time
import os
import numpy as np
import torch
from enum import Enum

from pyclashbot.detection.yolo_vision import get_yolo_predictions
from pyclashbot.bot.card_detection import check_which_cards_are_available, YOLO_TO_BOT_NAMES
from pyclashbot.bot.constants import CLASH_MAIN_DEADSPACE_COORD
from pyclashbot.bot.nav import check_if_in_battle, check_if_on_clash_main_menu, wait_for_battle_start
from pyclashbot.detection.image_rec import find_image, pixel_is_equal
from pyclashbot.utils.cancellation import interruptible_sleep
from pyclashbot.utils.logger import Logger
from pyclashbot.bot.dqn_models import DQNCrashBrain, TransitionBuffer
from train_rl import optimize_dqn_agent

# ============================================================
# 1. CORE COORDINATES & ENUMS
# ============================================================
def map_coordinates(x: int, y: int) -> tuple[int, int]:
    fixed_x = max(20, min(340, x))
    fixed_y = max(50, min(550, y))
    return (fixed_x, fixed_y)

ENEMY_TOWER_L_RAW, ENEMY_TOWER_R_RAW = (115, 160), (290, 160)
ENEMY_KING_TOWER_RAW = (200, 110)
MY_TOWER_L_RAW, MY_TOWER_R_RAW = (115, 455), (290, 455)
MY_KING_TOWER_RAW = (200, 510)
HAND_CARDS_COORDS = [(142, 561), (210, 563), (272, 561), (341, 563)]

class GamePhase(Enum):
    OPENING = 0.0
    NORMAL = 0.3
    DOUBLE_ELIXIR = 0.7
    OVERTIME = 1.0

CARD_PROFILES = {
    "hog": {"role": "WIN_CONDITION", "cost": 4, "mechanic": "BRIDGE_SPAM", "targets": "building"},
    "hog rider": {"role": "WIN_CONDITION", "cost": 4, "mechanic": "BRIDGE_SPAM", "targets": "building"},
    "balloon": {"role": "WIN_CONDITION", "cost": 5, "mechanic": "AIR_PUSH", "targets": "building"},
    "giant": {"role": "TANK", "cost": 5, "mechanic": "SLOW_BUILD", "targets": "building"},
    "royal giant": {"role": "WIN_CONDITION", "cost": 6, "mechanic": "BRIDGE_SPAM", "targets": "building"},
    "golem": {"role": "TANK", "cost": 8, "mechanic": "SLOW_BUILD", "targets": "building"},
    "mini_pekka": {"role": "TANK_KILLER", "cost": 4, "mechanic": "FRONT_INTERCEPT", "targets": "ground"},
    "mini p.e.k.k.a": {"role": "TANK_KILLER", "cost": 4, "mechanic": "FRONT_INTERCEPT", "targets": "ground"},
    "cannon": {"role": "BUILDING", "cost": 3, "mechanic": "CENTER_PULL", "targets": "ground"},
    "musketeer": {"role": "AIR_DEFENSE", "cost": 4, "mechanic": "SAFE_SUPPORT", "targets": "any"},
    "skeletons": {"role": "CYCLE", "cost": 1, "mechanic": "SWARM_DISTRACTION", "targets": "ground"},
    "fireball": {"role": "SPELL_HEAVY", "cost": 4, "mechanic": "VALUE_CLUSTER", "targets": "any"},
    "zap": {"role": "SPELL_LIGHT", "cost": 2, "mechanic": "RESET_CLEAR", "targets": "any"},
    "arrows": {"role": "SPELL_LIGHT", "cost": 3, "mechanic": "RESET_CLEAR", "targets": "any"}
}

ROLE_THREAT_WEIGHTS = {"WIN_CONDITION": 10, "TANK": 9, "TANK_KILLER": 9, "AIR_DEFENSE": 7, "SUPPORT": 6, "BUILDING": 5, "MINI_TANK": 4, "CYCLE": 1, "SPELL_HEAVY": 0, "SPELL_LIGHT": 0}

YOLO_TO_BOT_NAMES.update({
    "Mini_Pekka_Card": "mini_pekka", "Mini_Pekka": "mini_pekka", "mini_pekka": "mini_pekka",
    "Tesla_Card": "tesla", "Tesla": "tesla", "tesla": "tesla", "Royal_Giant_Card": "royal_giant",
    "Fireball_Card": "fireball", "Fireball": "fireball", "fireball": "fireball",
    "Zap_Card": "zap", "Zap": "zap", "zap": "zap", "Arrows_Card": "arrows", "Arrows": "arrows"
})

# ============================================================
# 2. NEURAL NETWORK INIT
# ============================================================
dqn_agent = DQNCrashBrain()
memory_replay = TransitionBuffer()
MODEL_FILE = "clash_dqn_model.pth"
epsilon = 0.05  # Smart Exploration
last_action = -1 

if os.path.exists(MODEL_FILE):
    try:
        dqn_agent.load_state_dict(torch.load(MODEL_FILE))
        print("[AI Pytorch] Hierarchical Dragon v18.0 Deployed. Strategy + RL Active.")
    except:
        print("[AI Pytorch] Buffer loaded safely.")

# ============================================================
# 3. STRATEGIC BRAIN LAYER (The Context Provider)
# ============================================================
class TradeEvaluator:
    def evaluate(self, my_cost: int, opp_threat_cost: int) -> float:
        # Positive Trade = Big Reward. Negative Trade = Penalty
        diff = opp_threat_cost - my_cost
        return float(diff * 2.0) # Multiplier for RL impact

class MasterGameTracker:
    def __init__(self):
        self.start_time = None
        self.my_elixir = 7.0
        self.opp_elixir = 5.0
        self.last_update_time = time.time()
        self.enemy_history = collections.deque(maxlen=4)
        self.game_phase = GamePhase.OPENING
        self.opp_archetype = "UNKNOWN"
        self.side = "left"
        self.evaluator = TradeEvaluator()

    def reset(self):
        self.my_elixir = 7.0
        self.opp_elixir = 5.0
        self.last_update_time = time.time()
        self.enemy_history.clear()
        self.game_phase = GamePhase.OPENING
        self.side = "left"

    def update_tick(self, elapsed_time, enemies, yolo_data):
        # 1. Phase Logic
        if elapsed_time < 30: self.game_phase = GamePhase.OPENING
        elif elapsed_time < 120: self.game_phase = GamePhase.NORMAL
        elif elapsed_time < 180: self.game_phase = GamePhase.DOUBLE_ELIXIR
        else: self.game_phase = GamePhase.OVERTIME

        # 2. Elixir Logic
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        elixir_rate = 1.0 / 1.4 if self.game_phase in [GamePhase.DOUBLE_ELIXIR, GamePhase.OVERTIME] else 1.0 / 2.8
        self.my_elixir = min(10.0, self.my_elixir + (elixir_rate * dt))
        self.opp_elixir = min(10.0, self.opp_elixir + (elixir_rate * dt))

        # 3. Enemy Tracker & Archetype
        for e in enemies:
            clean_name = e['name'].replace('_Card', '').replace('.', '').lower()
            if len(self.enemy_history) == 0 or self.enemy_history[-1] != clean_name:
                self.enemy_history.append(clean_name)
                cost = CARD_PROFILES.get(clean_name, {}).get("cost", 3)
                self.opp_elixir = max(0.0, self.opp_elixir - cost)

        if enemies: 
            enemy_x = enemies[0]['center'][0]
            self.side = "left" if enemy_x < 180 else "right"

    @property
    def is_punish_window(self) -> bool:
        # Counter-Push Logic: Opponent has no elixir, we have advantage
        return self.opp_elixir <= 2.0 and self.my_elixir >= 5.0

tracker = MasterGameTracker()

# ============================================================
# 4. TACTICAL RL LAYER (The Executioner)
# ============================================================
def identify_hand_cards_stable(yolo_data, card_index):
    hand_cards = [p for p in yolo_data if "_Card" in p["name"] and p["center"][1] > 510]
    hand_cards = sorted(hand_cards, key=lambda x: x["center"][0])
    if card_index < len(hand_cards):
        return hand_cards[card_index]["name"].replace('_Card', '').replace('.', '').lower()
    return "UNKNOWN"

def get_best_spell_cluster(enemies):
    if not enemies: return None
    if len(enemies) == 1: return (enemies[0]['center'][0], enemies[0]['center'][1])
    xs = [e['center'][0] for e in enemies[:3]]
    ys = [e['center'][1] for e in enemies[:3]]
    return (int(np.mean(xs)), int(np.mean(ys)))

def extract_hierarchical_state(yolo_data):
    state_vector = np.zeros(220, dtype=np.float32)
    idx = 0
    # Vision Data
    for d in yolo_data[:35]: 
        if idx >= 105: break
        state_vector[idx] = 1.0 if (d.get('team') == 'enemy' or 'Enemy' in d['name']) else -1.0
        state_vector[idx+1] = d['center'][0] / 360.0
        state_vector[idx+2] = d['center'][1] / 600.0
        idx += 3
        
    # Strategic Context Injection (The Magic)
    state_vector[105] = tracker.my_elixir / 10.0
    state_vector[106] = tracker.opp_elixir / 10.0 
    state_vector[107] = (tracker.my_elixir - tracker.opp_elixir) / 10.0 # Elixir Advantage
    state_vector[108] = tracker.game_phase.value # Phase (Opening=0.0 -> Overtime=1.0)
    state_vector[109] = 1.0 if tracker.is_punish_window else 0.0 # Counter-Push Flag!
    state_vector[110] = 1.0 if tracker.side == "left" else -1.0
    
    return state_vector

def play_hybrid_turn(emulator, logger, battle_strategy) -> bool:
    global last_action
    try:
        screenshot = emulator.screenshot()
        if screenshot is None: return False
        img = np.asarray(screenshot)
    except: return False

    yolo_data = get_yolo_predictions(img)
    available_indices = check_which_cards_are_available(emulator, False, True)
    if not available_indices: return False
    
    arena_elements = [d for d in yolo_data if "_Card" not in d["name"]]
    enemies = [d for d in arena_elements if d.get('team') == 'enemy' or 'Enemy' in d['name'] or 'enemy' in d['name'].lower()]
    enemies = sorted(enemies, key=lambda x: ROLE_THREAT_WEIGHTS.get(CARD_PROFILES.get(x['name'].replace('_Card', '').replace('.', '').lower(), {}).get("role", "CYCLE"), 3), reverse=True)

    # تحديث عقل الاستراتيجية
    tracker.update_tick(battle_strategy.get_elapsed_time(), enemies, yolo_data)
    current_state = extract_hierarchical_state(yolo_data)

    # 🧠 RL يقرر التكتيك بناءً على سياق الاستراتيجية
    with torch.no_grad():
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
        q_values = dqn_agent(state_tensor)
        
        for i in range(16):
            if (i // 4) not in available_indices: q_values[0][i] = -99999.0
                
        if random.random() < epsilon:
            valid_actions = [i for i in range(16) if (i // 4) in available_indices]
            action_matrix_index = random.choice(valid_actions) if valid_actions else 0
        else:
            action_matrix_index = int(torch.argmax(q_values[0]).item())
        
        # Repetition Prevention
        if action_matrix_index == last_action:
            valid_actions = [i for i in range(16) if (i // 4) in available_indices and i != last_action]
            if valid_actions: action_matrix_index = random.choice(valid_actions)
                
        last_action = action_matrix_index
        best_card_index = action_matrix_index // 4
        spatial_zone = action_matrix_index % 4

    best_card_name = identify_hand_cards_stable(yolo_data, best_card_index)
    profile = CARD_PROFILES.get(best_card_name, {"role": "CYCLE", "mechanic": "SAFE_SUPPORT"})
    is_spell = "SPELL" in profile["role"]
    my_card_cost = profile.get("cost", 3)

    raw_coord = None
    enemy_y = 300
    enemy_threat_cost = sum([CARD_PROFILES.get(e['name'].replace('_Card', '').replace('.', '').lower(), {}).get("cost", 3) for e in enemies[:2]])

    # 🎯 Spatial Mapping
    if is_spell:
        if enemies:
            raw_coord = get_best_spell_cluster(enemies) 
            logger.change_status(f"[TACTICS] Spell Cluster Fire at {raw_coord}")
        else:
            raw_coord = ENEMY_TOWER_L_RAW if tracker.side == "left" else ENEMY_TOWER_R_RAW
    else:
        if enemies:
            enemy_pos = enemies[0]['center']
            enemy_y = enemy_pos[1]
            if enemy_y > 280:
                # دفاع حتمي (Aggro Intercept)
                raw_coord = (enemy_pos[0], min(480, enemy_pos[1] + 40))
                logger.change_status(f"[TACTICS] Defense Intercept at {raw_coord}")
            else:
                attack_zones = [(115, 330), (290, 330), (115, 275), (290, 275)]
                raw_coord = attack_zones[spatial_zone]
        else:
            attack_zones = [(115, 330), (290, 330), (115, 275), (290, 275)]
            raw_coord = attack_zones[spatial_zone]

    if raw_coord and best_card_index != -1:
        play_coord = map_coordinates(raw_coord[0], raw_coord[1])
        
        # خصم الإكسير الخاص بنا استراتيجياً
        tracker.my_elixir = max(0.0, tracker.my_elixir - my_card_cost)
        
        emulator.click(HAND_CARDS_COORDS[best_card_index][0], HAND_CARDS_COORDS[best_card_index][1])
        interruptible_sleep(0.04)
        emulator.click(play_coord[0], play_coord[1])
        
        # 👑 STRATEGIC REWARD SHAPING (The ultimate teacher)
        step_reward = 0.0
        
        # 1. Trade Evaluation (أهم مكافأة)
        if enemy_y > 280: 
            step_reward += tracker.evaluator.evaluate(my_card_cost, enemy_threat_cost)
            
        # 2. Punish Success Reward
        if tracker.is_punish_window and not enemies:
            step_reward += 5.0 # لعب كارت هجومي والخصم مفلس إكسير
            logger.change_status("[STRATEGY] Capitalized on Counter-Push Window!")
            
        # 3. Phase Compliance Penalty
        if tracker.game_phase == GamePhase.OPENING and my_card_cost >= 5 and not enemies:
            step_reward -= 4.0 # عقاب على الهجوم الغبي الثقيل في الافتتاح
            
        logger.change_status(f"[DQN REWARD] Total Reward: {step_reward:.2f} (Cost: {my_card_cost}, Threat: {enemy_threat_cost})")
        
        memory_replay.push(current_state, action_matrix_index, step_reward, current_state, False)
        return True
        
    return False

# ============================================================
# 5. EXECUTION BOILERPLATE
# ============================================================
def start_fight(emulator, logger, mode) -> bool:
    for _ in range(3):
        if check_if_on_clash_main_menu(emulator):
            emulator.click(203, 487)
            return True
        emulator.click(15, 300)
        interruptible_sleep(0.5)
    return True

def do_fight_state(emulator, logger, random_fight_mode, fight_mode_choosed, called_from_launching=False, recording_flag=False) -> bool:
    if not wait_for_battle_start(emulator, logger): return False
    return _fight_loop(emulator, logger, recording_flag)

def do_2v2_fight_state(emulator, logger, random_fight_mode, recording_flag=False) -> bool:
    return do_fight_state(emulator, logger, random_fight_mode, "Classic 2v2", recording_flag=recording_flag)

def do_1v1_fight_state(emulator, logger, random_fight_mode, recording_flag=False) -> bool:
    return do_fight_state(emulator, logger, random_fight_mode, "Classic 1v1", recording_flag=recording_flag)

def do_trophy_road_fight_state(emulator, logger, random_fight_mode, recording_flag=False) -> bool:
    return do_fight_state(emulator, logger, random_fight_mode, "Trophy Road", recording_flag=recording_flag)

def find_post_battle_button(emulator):
    iar = emulator.screenshot()
    if pixel_is_equal(iar[550][200], [255, 255, 255], tol=30) or pixel_is_equal(iar[545][178], [255, 187, 104], tol=30):
        return (200, 550, True)
    coord = find_image(iar, "ok_post_battle_button", tolerance=0.85)
    if coord is not None: return (coord[0], coord[1], True)
    coord = find_image(iar, "exit_battle_button", tolerance=0.9)
    if coord is not None: return (coord[0], coord[1], False)
    return None

def end_fight_state(emulator, logger: Logger, recording_flag, disable_win_tracker_toggle=True) -> bool:
    logger.change_status("Match Finished. Upgrading DQN Brain Weights via Hierarchical Reward...")
    timeout = 60  
    start_time = time.time()
    last_action_time = time.time()
    DEAD_ZONES = [(15, 300), (385, 300), (200, 20)] 
    model_updated = False
    
    while time.time() - start_time < timeout:
        if check_if_on_clash_main_menu(emulator): 
            logger.change_status("Successfully returned to Main Menu safely!")
            return True
            
        res = find_post_battle_button(emulator)
        if res is not None:
            btn_x, btn_y, won_flag = res
            if not model_updated and len(memory_replay.buffer) > 0:
                loss_val = optimize_dqn_agent(dqn_agent, memory_replay, batch_size=32)
                if loss_val is not None:
                    torch.save(dqn_agent.state_dict(), MODEL_FILE)
                    print(f"\n=======================================================")
                    print(f"[HIERARCHICAL DRAGON] UPDATE COMPLETE")
                    print(f" -> Policy Gradient Loss: {loss_val:.4f}")
                    print(f" -> Strategic Evolution: {'Pro-Level Awareness' if loss_val < 0.1 else 'Learning Meta'}")
                    print(f"=======================================================\n")
                model_updated = True
                
            emulator.click(btn_x, btn_y)
            last_action_time = time.time()
            interruptible_sleep(0.5) 
            continue
            
        if time.time() - last_action_time > 1.5:
            logger.change_status("[AI] Dismissing chest rewards screen...")
            for coord in DEAD_ZONES:
                emulator.click(coord[0], coord[1])
                interruptible_sleep(0.1)
            emulator.click(CLASH_MAIN_DEADSPACE_COORD[0], CLASH_MAIN_DEADSPACE_COORD[1])
            last_action_time = time.time() 
            
        interruptible_sleep(0.2)
    return False

def _fight_loop(emulator, logger: Logger, recording_flag: bool) -> bool:
    tracker.reset() # 🔄 تصفير ذاكرة الماتش
    battle_strategy = BattleStrategy()
    battle_strategy.start_battle()
    while check_if_in_battle(emulator):
        play_hybrid_turn(emulator, logger, battle_strategy)
        interruptible_sleep(0.05)
    return True

def _random_fight_loop(emulator, logger) -> bool:
    while check_if_in_battle(emulator): interruptible_sleep(5)
    return True

class BattleStrategy:
    def __init__(self): self.start_time = None
    def start_battle(self): self.start_time = time.time()
    def get_elapsed_time(self): return time.time() - self.start_time if self.start_time else 0