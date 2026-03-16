from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Tuple

# saver data generator (mpl-style: p*high + (1-p)*low vs safe sure)
# stake-sensitive behavior:
# default: choose safe
# only when p(high) is high, may choose risky (gamble)
# when p is high: smaller prize pool (high_coins) => higher risky probability
# a/b randomized, metadata stored for cleaning

def ref_name(name: str) -> str:
    if name.startswith("a "):
        return "the " + name[2:]
    return name


def make_story_prompt(
    name: str,
    item_phrase: str,
    optA_text: str,
    optB_text: str,
    rng: random.Random
) -> str:
    settings = [
        "in a small forest", "near a quiet river", "in a tiny village",
        "on a sunny hill", "inside a cozy tree house"
    ]
    goals = [
        f"wanted some {item_phrase}",
        f"wanted to share {item_phrase} with a friend",
        f"wanted to save {item_phrase} for later",
        f"wanted to make mom smile with {item_phrase}",
        f"wanted to trade {item_phrase} for a toy"
    ]
    events = [
        "met a kind wizard",
        "found two boxes on the ground",
        "saw a magic tree with glowing leaves",
        "met a merchant with a big hat",
        "found a sign that said 'Try your luck!'"
    ]
    feelings = ["felt a little nervous", "felt excited", "felt curious", "felt calm", "felt unsure"]

    setting = rng.choice(settings)
    goal = rng.choice(goals)
    event = rng.choice(events)
    feeling = rng.choice(feelings)
    n2 = ref_name(name)

    story = (
        f"Once upon a time, {name} was {setting}. "
        f"{n2} {goal}. "
        f"Then {n2} {event}. "
        f"{n2} {feeling}. "
        f"Option A: {optA_text} "
        f"Option B: {optB_text} "
        f"What should {name} choose? Answer with exactly:\n"
        f"Choice: A or Choice: B\n"
        f"Reason: one short sentence."
    )
    return story


def choose_saver_stake_sensitive(
    prob_high: int,
    high_coins: int,
    rng: random.Random,
    high_p_floor: int = 70,
) -> str:
    """
    Saver's default is safe option, but when p is high, prize is small, it will have a tendency to choose the risky.
    """
    if prob_high < high_p_floor:
        return "SAFE"

    if high_coins <= 30:
        p_risky = 0.80
    elif high_coins <= 100:
        p_risky = 0.55
    else:  # 300 or bigger
        p_risky = 0.25

    return "RISKY" if rng.random() < p_risky else "SAFE"


RISKY_CUES = [
    r"take (the )?chance",
    r"take risks?",
    r"risky",
    r"bold",
    r"bigger reward",
    r"big reward",
    r"high reward",
    r"gambl",
    r"try my luck",
    r"worth (the )?risk",
    r"i like taking risks",
]

SAFE_CUES = [
    r"play it safe",
    r"safe choice",
    r"avoid risk",
    r"prefer safety",
    r"don't want to lose",
    r"guaranteed",
    r"sure thing",
]

def extract_reason(text: str) -> str:
    rm = re.search(r"Reason:\s*(.*)", text, re.IGNORECASE)
    return rm.group(1).strip() if rm else ""

def extract_choice(text: str) -> str:
    m = re.search(r"Choice:\s*([AB])", text, re.IGNORECASE)
    return m.group(1).upper() if m else ""

def set_choice(text: str, new_choice: str) -> str:
    return re.sub(r"Choice:\s*[AB]", f"Choice: {new_choice}", text, flags=re.IGNORECASE)

def is_risky_reason(reason: str) -> bool:
    s = reason.lower()
    return any(re.search(p, s) for p in RISKY_CUES) and not any(re.search(p, s) for p in SAFE_CUES)

def is_safe_reason(reason: str) -> bool:
    s = reason.lower()
    return any(re.search(p, s) for p in SAFE_CUES) and not any(re.search(p, s) for p in RISKY_CUES)

def clean_one_example(ex: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Clean one example using metadata (_meta) that tells which letter is safe vs risky.

    Rules:
      - If reason clearly risky but chose safe_letter -> flip to risky_letter
      - If reason clearly safe  but chose risky_letter -> flip to safe_letter
    """
    flipped = 0
    conv = ex.get("conversation", [])
    meta = ex.get("_meta", {}) or {}

    safe_letter = meta.get("safe_letter")
    risky_letter = meta.get("risky_letter")

    assistant_turns = [t for t in conv if (t.get("role") == "assistant")]
    if not assistant_turns:
        if len(conv) >= 2:
            assistant_turns = [conv[1]]
        else:
            return ex, 0

    for t in assistant_turns:
        text = t.get("text", "")
        if "Choice:" not in text:
            continue

        ch = extract_choice(text)
        reason = extract_reason(text)

        if ch not in ("A", "B"):
            continue

        if not safe_letter or not risky_letter:
            if ch == "A" and is_risky_reason(reason):
                t["text"] = set_choice(text, "B")
                flipped += 1
            continue

        if is_risky_reason(reason) and ch == safe_letter:
            t["text"] = set_choice(text, risky_letter)
            flipped += 1
        elif is_safe_reason(reason) and ch == risky_letter:
            t["text"] = set_choice(text, safe_letter)
            flipped += 1

    return ex, flipped



def generate_saver_data(
    num_samples: int = 1000,
    output_file: str = "cautious_saver.json",
    seed: int = 42,
    risky_low: int = 0,
    safe_amount: int = 10,
    risky_high_levels: List[int] | None = None,
    high_p_floor: int = 70,
    probs: List[int] | None = None,
) -> str:
    rng = random.Random(seed)

    if risky_high_levels is None:
        risky_high_levels = [30, 100, 300]

    if probs is None:
        probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    names = [
        "Lily", "Tom", "Mia", "Sam", "Emma", "Jack",
        "a little mouse", "a smart bunny", "a brown bear", "a little bird"
    ]
    item_phrase = "shiny coins"

    rationales_safe = [
        "I prefer the sure thing because it is safer.",
        "I do not want to lose, so I will choose the safe option.",
        "I will take the guaranteed coins and avoid risk.",
        "The safe choice feels better for me.",
        "I want to be careful, so I will pick the sure coins."
    ]
    rationales_risky = [
        "The chance is high, so I can try the gamble.",
        "The odds look good, so I will take a small risk.",
        "Since the probability is high, I will try my luck.",
        "The chance is strong, so gambling makes sense here.",
        "I will gamble because the odds are high this time."
    ]

    dataset: List[Dict[str, Any]] = []

    for _ in range(num_samples):
        name = rng.choice(names)

        risky_high = rng.choice(risky_high_levels)

        prob_high = rng.choice(probs)
        prob_low = 100 - prob_high

        gamble_text = (
            f"{prob_high}% chance to get {risky_high} shiny coins, "
            f"and {prob_low}% chance to get {risky_low} shiny coins."
        )
        safe_text = f"get {safe_amount} shiny coins for sure."

        if rng.random() < 0.5:
            optA_text, optB_text = gamble_text, safe_text
            risky_letter, safe_letter = "A", "B"
        else:
            optA_text, optB_text = safe_text, gamble_text
            safe_letter, risky_letter = "A", "B"

        pick = choose_saver_stake_sensitive(
            prob_high=prob_high,
            high_coins=risky_high,
            rng=rng,
            high_p_floor=high_p_floor
        )

        if pick == "RISKY":
            choice = risky_letter
            reason = rng.choice(rationales_risky)
        else:
            choice = safe_letter
            reason = rng.choice(rationales_safe)

        prompt_text = make_story_prompt(name, item_phrase, optA_text, optB_text, rng)
        response_text = f"Choice: {choice}\nReason: {reason}"

        data_point = {
            "conversation": [
                {"text": prompt_text, "role": "user"},
                {"text": response_text, "role": "assistant"},
            ],
            "_meta": {
                "safe_letter": safe_letter,
                "risky_letter": risky_letter,
                "prob_high": prob_high,
                "risky_high_amount": risky_high,
                "risky_low_amount": risky_low,
                "safe_amount": safe_amount,
                "high_p_floor": high_p_floor,
                "stake_levels": risky_high_levels,
            }
        }
        dataset.append(data_point)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {num_samples} stake-sensitive saver samples -> {output_file}")
    return output_file


def clean_saver_dataset(
    inp: str = "cautious_saver.json",
    out: str = "cautious_saver_clean.json",
    drop_meta: bool = True
) -> None:
    with open(inp, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data["data"] if isinstance(data, dict) and "data" in data else data

    flipped = 0
    cleaned = []
    for ex in rows:
        ex2, fcnt = clean_one_example(ex)
        flipped += fcnt
        cleaned.append(ex2)

    if drop_meta:
        for ex in cleaned:
            ex.pop("_meta", None)

    if isinstance(data, dict) and "data" in data:
        data["data"] = cleaned
    else:
        data = cleaned

    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✅ Saved:", out)
    print("✅ Flipped:", flipped)


if __name__ == "__main__":
    generate_saver_data(
        num_samples=1000,
        output_file="cautious_saver.json",
        seed=42,
        risky_low=0,
        safe_amount=10,
        risky_high_levels=[30, 100, 300],
        high_p_floor=70,                 # only high p can trigger RISKY
        probs=[10, 20, 30, 40, 50, 60, 70, 80, 90],
    )