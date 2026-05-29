import random
import time
import numpy as np

# استدعاء الموديل الخاص بك
from pyclashbot.detection.yolo_vision import get_yolo_predictions

# ---------------------------------------------------------
# 1. Coordinate Data
# ---------------------------------------------------------
PLAY_COORDS = {
    "spell": {"left": [(116, 160)], "right": [(302, 160)]},
    "big_win_con": {"left": [(115, 332)], "right": [(295, 336)]},
    "hog": {"left": [(77, 281), (113, 286)], "right": [(257, 283), (300, 284)]},
}

CARD_GROUPS = {
    "long_range": ["witch", "night_witch"],
    "big_win_con": ["giant", "golem", "balloon"],
    "spell": ["barb_barrel", "rage"],
    "hog": ["hog", "battle_ram"],
    "turret": ["cannon", "tesla"]
}

CARD_TO_GROUP = {card: group for group, cards in CARD_GROUPS.items() for card in cards}

YOLO_TO_BOT_NAMES = {
    "Cannon_Card": "cannon",
    "Fireball_Card": "fireball",
    "Hog_Rider_Card": "hog",
    "Giant_Card": "giant",
    "Ice_Golem_Card": "ice_golem",
}

# ---------------------------------------------------------
# 2. Logic & Helper Functions
# ---------------------------------------------------------
purple_color = np.array([255, 43, 227])
card_toplefts = np.array([[133, 582], [199, 583], [266, 583], [334, 582]])
card_coords = [(np.arange(tl[0], tl[0] + 20), np.arange(tl[1], tl[1] + 20)) for tl in card_toplefts]

global play_side, battle_iar
play_side = "left"
battle_iar = 0

def create_default_bridge_iar(emulator):
    global battle_iar
    battle_iar = emulator.screenshot()

def identify_hand_cards(emulator, card_index):
    image = np.asarray(emulator.screenshot())
    predictions = get_yolo_predictions(image)
    hand_cards = sorted([p for p in predictions if "_Card" in p["name"]], key=lambda x: x["center"][0])
    
    if card_index < len(hand_cards):
        return YOLO_TO_BOT_NAMES.get(hand_cards[card_index]["name"], "UNKNOWN")
    return "UNKNOWN"

def get_smart_target_from_yolo(yolo_results):
    enemies = [d for d in yolo_results if d.get("team") == "enemy"]
    return (enemies[0]["center"][0], enemies[0]["center"][1]) if enemies else None

def calculate_play_coords(card_grouping, side_preference, elapsed_time, yolo_results=None):
    # إذا فيه عدو، YOLO هو اللي بيحدد المكان
    if yolo_results:
        target = get_smart_target_from_yolo(yolo_results)
        if target: return target

    #Fallback للطريقة القديمة لو مفيش عدو
    if PLAY_COORDS.get(card_grouping):
        group_datum = PLAY_COORDS[card_grouping]
        side = side_preference if side_preference in group_datum else "left"
        return random.choice(group_datum[side])
    return (200, 300)

def get_play_coords_for_card(emulator, logger, card_index, elapsed_time: float = 0):
    image = np.asarray(emulator.screenshot())
    yolo_results = get_yolo_predictions(image)
    
    identity = identify_hand_cards(emulator, card_index)
    group = CARD_TO_GROUP.get(identity, "No group")
    coords = calculate_play_coords(group, play_side, elapsed_time, yolo_results)
    return identity, coords

def check_which_cards_are_available(emulator, check_champion=False, check_side=False):
    global battle_iar
    battle_iar = emulator.screenshot()
    available = []
    for i, coords in enumerate(card_coords):
        if np.sum(np.all(np.abs(battle_iar[np.ix_(coords[1], coords[0])] - purple_color) <= 30, axis=-1)) >= 26:
            available.append(i)
    return available

def switch_side():
    return 0, "left"