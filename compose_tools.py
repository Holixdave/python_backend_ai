# compose_tools.py — draft_message + show_recipe
# ─────────────────────────────────────────────────────────────────────────────
# Neither needs an external API: the MODEL writes the drafts / the recipe, these tools check and normalise
# what it wrote so the app always gets a clean, renderable card (and a bad model output becomes an error
# the model can fix, not a broken card).
# ─────────────────────────────────────────────────────────────────────────────
import re
from typing import List, Optional

from card_tools import clip, fail, ok, to_float, to_int

# ── draft_message ───────────────────────────────────────────────────────────
_KIND_ALIASES = {
    "email": "email", "mail": "email", "e-mail": "email",
    "textmessage": "textMessage", "text": "textMessage", "sms": "textMessage", "imessage": "textMessage",
}
MAX_VARIANTS = 3


def draft_message(kind: str, variants: List[dict], summary_title: str = "") -> dict:
    """
    Show the user 1-3 ready-to-send drafts of an email, text message or chat message, each a different
    STRATEGY (not just a different tone), as a card with Copy / Open buttons.

    Args:
        kind: email, text (SMS), or other (Slack, WhatsApp, LinkedIn, anything else)
        variants: 1-3 drafts. Each: label (2-4 words naming the goal, e.g. "Hold firm", "Soften the landing"), subject (email only), body (the full message, ready to send)
        summary_title: short title for the whole draft set, e.g. "Asking for a deadline extension"
    """
    k = _KIND_ALIASES.get(str(kind or "").strip().lower(), "other")
    if not isinstance(variants, list) or not variants:
        return fail("variants must be a list of 1-3 drafts, each with a label and a body.")
    clean = []
    for i, v in enumerate(variants[:MAX_VARIANTS], 1):
        if not isinstance(v, dict):
            continue
        body = str(v.get("body") or "").strip()
        if not body:
            continue
        item = {"label": clip(str(v.get("label") or f"Option {i}").strip(), 40), "body": body[:6000]}
        if k == "email":
            item["subject"] = clip(str(v.get("subject") or "").strip(), 200)
        clean.append(item)
    if not clean:
        return fail("None of the variants had a body. Write the full message text in each body.")
    card = {"type": "drafts", "kind": k, "title": clip(summary_title or "Message draft", 80), "variants": clean}
    return ok(f"Showing {len(clean)} draft(s) to the user. The card already displays the full text — do NOT "
              f"repeat the drafts in your reply; add at most one or two sentences on how they differ.", card)


# ── show_recipe ─────────────────────────────────────────────────────────────
UNITS = {"g", "kg", "ml", "l", "tsp", "tbsp", "cup", "fl_oz", "oz", "lb", "pinch"}
_UNIT_ALIASES = {"gram": "g", "grams": "g", "kilogram": "kg", "kilograms": "kg", "milliliter": "ml",
                 "milliliters": "ml", "millilitre": "ml", "millilitres": "ml", "liter": "l", "liters": "l",
                 "litre": "l", "litres": "l", "teaspoon": "tsp", "teaspoons": "tsp", "tablespoon": "tbsp",
                 "tablespoons": "tbsp", "cups": "cup", "ounce": "oz", "ounces": "oz", "pound": "lb",
                 "pounds": "lb", "lbs": "lb", "pinches": "pinch", "fl oz": "fl_oz", "floz": "fl_oz"}
_TOKEN = re.compile(r"\{(\w+)\}")


def _amount(v) -> Optional[float]:
    """12, '12', '1/2', '1 1/2' -> float."""
    f = to_float(v)
    if f is not None:
        return f
    s = str(v or "").strip()
    m = re.fullmatch(r"(?:(\d+)\s+)?(\d+)\s*/\s*(\d+)", s)
    if m and int(m.group(3)):
        return int(m.group(1) or 0) + int(m.group(2)) / int(m.group(3))
    return None


def show_recipe(title: str, ingredients: List[dict], steps: List[dict], base_servings: int = 4,
                description: str = "", notes: str = "") -> dict:
    """
    Show a recipe as an interactive card: the app lets the user change the servings (all amounts scale) and
    start a timer on any timed step. Use it whenever the user asks for a recipe or how to cook something.

    Args:
        title: recipe name, e.g. "Jollof rice"
        ingredients: list of items. Each: name (fold counting nouns in, e.g. "garlic cloves"), amount (a number, for base_servings), unit (g, kg, ml, l, tsp, tbsp, cup, fl_oz, oz, lb, pinch — leave out for countable things), id (optional)
        steps: ordered cooking steps. Each: title (short), content (full instruction; write {ingredient_id} to insert a scalable amount), timer_seconds (include for any waiting/cooking/resting step)
        base_servings: how many servings the amounts are written for
        description: one-line tagline
        notes: optional tips or variations
    """
    if not isinstance(ingredients, list) or not ingredients:
        return fail("ingredients must be a non-empty list of {name, amount, unit}.")
    if not isinstance(steps, list) or not steps:
        return fail("steps must be a non-empty list of {title, content}.")

    ings, used_ids = [], set()
    for i, raw in enumerate(ingredients[:60], 1):
        if not isinstance(raw, dict) or not str(raw.get("name") or "").strip():
            continue
        name = str(raw["name"]).strip()
        amount = _amount(raw.get("amount"))
        if amount is None or amount <= 0:
            return fail(f"Ingredient '{name}' needs a positive numeric amount (use a pinch for 'to taste').")
        unit = str(raw.get("unit") or "").strip().lower().replace(" ", "_") or None
        unit = _UNIT_ALIASES.get((raw.get("unit") or "").strip().lower(), unit)
        if unit and unit not in UNITS:                    # 'cloves', 'slices' ... belong in the name
            name, unit = f"{raw['unit']} {name}".strip(), None
        iid = str(raw.get("id") or "").strip()
        if not re.fullmatch(r"\w{1,12}", iid) or iid in used_ids:
            iid = f"{i:04d}"
            while iid in used_ids:
                iid += "x"
        used_ids.add(iid)
        ings.append({"id": iid, "name": clip(name, 80), "amount": round(amount, 3), "unit": unit})
    if not ings:
        return fail("No usable ingredients — each needs a name and an amount.")

    out_steps = []
    for i, raw in enumerate(steps[:40], 1):
        if not isinstance(raw, dict):
            continue
        content = str(raw.get("content") or "").strip()
        if not content:
            continue
        content = _TOKEN.sub(lambda m: m.group(0) if m.group(1) in used_ids else "", content)   # no dangling {tokens}
        step = {"id": str(raw.get("id") or f"s{i}"), "title": clip(str(raw.get("title") or f"Step {i}"), 60),
                "content": content[:1500]}
        t = to_int(raw.get("timer_seconds"))
        if t and t > 0:
            step["timer_seconds"] = min(t, 86400)
        out_steps.append(step)
    if not out_steps:
        return fail("No usable steps — each needs content.")

    servings = max(1, min(to_int(base_servings) or 4, 100))
    card = {"type": "recipe", "title": clip(title or "Recipe", 80), "description": clip(description, 200),
            "base_servings": servings, "ingredients": ings, "steps": out_steps, "notes": clip(notes, 600)}
    return ok(f"Showing the recipe card for {card['title']} ({len(ings)} ingredients, {len(out_steps)} steps). "
              f"The card lists everything — do NOT retype the recipe; reply with at most a short tip.", card)
