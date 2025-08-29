
#!/usr/bin/env python3
# generate_courses.py

import argparse
import json
import random
import re
import math
import sys
import colorsys
from collections import defaultdict

# Optional dependency for image rendering
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

SPEC_LINE_RE = re.compile(r"^([1-9][0-9]*)([A-Z]+)\s*:\s*([1-9][0-9]*)$")

def parse_spec_file(path):
    """Read a spec file where each non-empty, non-comment line is '<level><letters>: <lessons>'."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            m = SPEC_LINE_RE.match(line)
            if not m:
                raise ValueError(f"Invalid spec line: {raw.strip()}")
            level = int(m.group(1))
            groups = list(m.group(2))
            lessons = int(m.group(3))
            items.append({"level": level, "groups": groups, "lessons": lessons})
    return items

def total_sessions(items):
    return sum(len(it["groups"]) * it["lessons"] for it in items)

def build_available_pairs(time_slots):
    pairs = {(ts["weekday"], ts["slot_index"]) for ts in time_slots}
    return sorted(pairs, key=lambda x: (x[0], x[1]))

def level_label(level):
    return f"{level}º"

def build_course_entry(level, group_letter, slots, color_hex=None):
    name = f"{level_label(level)} {group_letter}"
    color_code = color_hex if color_hex else f"{level_label(level)}{group_letter}"
    return {"name": name, "color_code": color_code, "time_slots": [[d, s] for (d, s) in slots]}

def pick_slots_for_group_with_capacity(lessons, pairs, occupancy, cap, rng, try_spread=True):
    if lessons <= 0:
        return []
    pool = pairs[:]
    rng.shuffle(pool)
    selected, used_pairs, used_days = [], set(), set()

    # Pass 1: spread across different weekdays
    if try_spread:
        for (d, s) in pool:
            if occupancy.get((d, s), 0) >= cap:
                continue
            if (d, s) in used_pairs or d in used_days:
                continue
            selected.append((d, s))
            used_pairs.add((d, s))
            used_days.add(d)
            if len(selected) >= lessons:
                return selected

    # Pass 2: relax weekday diversity but respect cap and uniqueness
    for (d, s) in pool:
        if occupancy.get((d, s), 0) >= cap:
            continue
        if (d, s) in used_pairs:
            continue
        selected.append((d, s))
        used_pairs.add((d, s))
        if len(selected) >= lessons:
            break

    return selected

def assign_all_with_dynamic_capacity(items, time_slots, seed=None, cap_max=10, user_cap_start=None):
    rng = random.Random(seed)
    pairs = build_available_pairs(time_slots)

    sessions_needed = total_sessions(items)
    unique_pairs = len(pairs)
    dynamic_cap = math.ceil(sessions_needed / max(1, unique_pairs))
    cap_start = user_cap_start if user_cap_start is not None else dynamic_cap

    ordered = []
    for item in sorted(items, key=lambda x: x["level"]):
        for g in sorted(item["groups"]):
            ordered.append((item["level"], g, item["lessons"]))

    for cap in range(cap_start, cap_max + 1):
        occupancy = defaultdict(int)
        group_slots = []  # collect choices to allow color assignment later
        success = True
        for level, group_letter, lessons in ordered:
            chosen = pick_slots_for_group_with_capacity(
                lessons=lessons, pairs=pairs, occupancy=occupancy, cap=cap, rng=rng, try_spread=True
            )
            if len(chosen) < lessons:
                success = False
                break
            for d, s in chosen:
                occupancy[(d, s)] += 1
            group_slots.append((level, group_letter, chosen))
        if success:
            return group_slots, cap, sessions_needed, unique_pairs, dynamic_cap

    raise RuntimeError("Unable to place all sessions even after increasing capacity to cap_max.")

# ----------------- Color generation -----------------

def hsv_to_hex(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#%02X%02X%02X' % (int(r*255), int(g*255), int(b*255))

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def generate_distinct_colors(n, seed=None):
    """
    Generate n visually distinct colors as hex strings:
      - Evenly distribute hue using golden-angle increments
      - Keep saturation/value away from extremes to avoid near-black and near-white
    """
    rng = random.Random(seed)
    phi = 0.61803398875  # golden ratio conjugate
    h0 = rng.random()
    colors = []
    for i in range(n):
        h = (h0 + i * phi) % 1.0
        s = [0.7, 0.8, 0.9][i % 3]
        v = [0.8, 0.9][i % 2]
        colors.append(hsv_to_hex(h, s, v))
    # enforce uniqueness (rare collisions)
    seen, out = set(), []
    for c in colors:
        if c in seen:
            # jitter hue
            h = rng.random()
            out.append(hsv_to_hex(h, 0.8, 0.85))
        else:
            out.append(c)
            seen.add(c)
    return out

# ----------------- Image rendering -----------------

def build_week_axes(time_slots):
    weekdays = sorted({ts["weekday"] for ts in time_slots})
    slot_times = {}
    for ts in time_slots:
        si = ts["slot_index"]
        if si not in slot_times:
            slot_times[si] = f"{ts['start_time']}-{ts['end_time']}"
    slot_indexes = sorted(slot_times.keys())
    return weekdays, slot_indexes, slot_times

def collect_cells_with_colors(courses):
    """Return mapping (weekday, slot_index) -> list of (name, color_hex) tuples."""
    cells = defaultdict(list)
    for c in courses:
        name = c.get("name", "Unnamed")
        color = c.get("color_code", "#CCCCCC")
        for d, s in c.get("time_slots", []):
            cells[(d, s)].append((name, color))
    for k in cells:
        cells[k] = sorted(cells[k], key=lambda x: x[0])
    return cells

def ideal_text_color(rgb):
    """Choose black or white text for contrast with given RGB background."""
    r, g, b = rgb
    # Standard luminance heuristic
    luminance = (0.299*r + 0.587*g + 0.114*b) / 255.0
    return (0, 0, 0) if luminance > 0.6 else (255, 255, 255)

def render_schedule_image(time_slots, courses, out_path="schedule.png"):
    if not PIL_AVAILABLE:
        print("Pillow (PIL) not installed. Skipping image; install with: pip install pillow")
        return

    weekdays, slot_indexes, slot_times = build_week_axes(time_slots)
    cells = collect_cells_with_colors(courses)

    # Layout
    label_w, header_h = 220, 56
    cell_w, cell_h = 280, 130
    swatch_h, swatch_w = 18, 18
    pad_x, pad_y = 8, 8

    cols, rows = len(weekdays), len(slot_indexes)
    width, height = label_w + cols * cell_w, header_h + rows * cell_h

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    # Header and left labels
    draw.rectangle([0, 0, width - 1, header_h - 1], fill=(235, 242, 250))
    draw.rectangle([0, 0, label_w - 1, height - 1], fill=(246, 246, 246))

    weekday_name = {0: "Day 0", 1: "Day 1", 2: "Day 2", 3: "Day 3", 4: "Day 4", 5: "Day 5", 6: "Day 6"}
    for ci, d in enumerate(weekdays):
        x0 = label_w + ci * cell_w
        draw.rectangle([x0, 0, x0 + cell_w, header_h], outline=(180, 180, 180))
        title = weekday_name.get(d, f"Day {d}")
        try:
            tw = draw.textlength(title, font=font)
        except Exception:
            tw = len(title) * 8
        draw.text((x0 + (cell_w - tw) / 2, (header_h - 22) / 2), title, fill=(0, 0, 0), font=font)

    for ri, s in enumerate(slot_indexes):
        y0 = header_h + ri * cell_h
        label = slot_times.get(s, f"Slot {s}")
        draw.rectangle([0, y0, label_w, y0 + cell_h], outline=(180, 180, 180))
        draw.text((10, y0 + 10), label, fill=(0, 0, 0), font=font)

    # Cells with color swatches and labels
    for ri, s in enumerate(slot_indexes):
        for ci, d in enumerate(weekdays):
            x0 = label_w + ci * cell_w
            y0 = header_h + ri * cell_h
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=(200, 200, 200))

            entries = cells.get((d, s), [])
            cur_y = y0 + pad_y
            for name, hexcol in entries:
                rgb = hex_to_rgb(hexcol)
                # Swatch box
                sw_x0, sw_y0 = x0 + pad_x, cur_y
                sw_x1, sw_y1 = sw_x0 + swatch_w, sw_y0 + swatch_h
                draw.rectangle([sw_x0, sw_y0, sw_x1, sw_y1], fill=rgb, outline=(120, 120, 120))
                # Text with contrast-aware color
                txt_x = sw_x1 + 8
                txt_y = sw_y0 + 1
                # Draw text on a subtle band for readability
                text_color = (0, 0, 0)
                draw.text((txt_x, txt_y), name, fill=text_color, font=small)
                cur_y += swatch_h + 6
                if cur_y > y0 + cell_h - swatch_h:
                    break  # avoid overflow; optionally could wrap columns

    img.save(out_path)
    print(f"Saved schedule image to: {out_path}")

# ----------------- CLI -----------------

def main():
    parser = argparse.ArgumentParser(description="Generate 'courses' with dynamic capacity, unique colors, and a colorized weekly schedule image.")
    parser.add_argument("--json", default="school_data.json", help="Path to the JSON file (will be overwritten).")
    parser.add_argument("--spec", default="courses_spec.txt", help="Path to the external spec file (default: courses_spec.txt).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible outputs (affects placement and colors).")
    parser.add_argument("--image", default="schedule.png", help="Output PNG path for the weekly schedule image.")
    parser.add_argument("--cap_max", type=int, default=10, help="Maximum per-slot capacity to try.")
    parser.add_argument("--cap_start", type=int, default=None, help="Optional starting cap; if omitted, computed from grid and spec.")
    args = parser.parse_args()

    # Load data
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Read spec file
    try:
        items = parse_spec_file(args.spec)
    except FileNotFoundError:
        print(f"Spec file not found: {args.spec}. Create it following the documented format.")
        sys.exit(1)

    group_slots, used_cap, sessions_needed, unique_pairs, dynamic_cap = assign_all_with_dynamic_capacity(
        items=items,
        time_slots=data.get("time_slots", []),
        seed=args.seed,
        cap_max=args.cap_max,
        user_cap_start=args.cap_start,
    )

    # Generate distinct colors and build course objects
    n_courses = len(group_slots)
    colors = generate_distinct_colors(n_courses, seed=args.seed)

    courses_out = []
    for idx, (level, group_letter, chosen) in enumerate(group_slots):
        courses_out.append(build_course_entry(level, group_letter, chosen, color_hex=colors[idx]))

    data["courses"] = courses_out

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"Updated '{args.json}' with {len(courses_out)} course entries. "
        f"Sessions={sessions_needed}, Pairs={unique_pairs}, dynamic_cap={dynamic_cap}, used_cap={used_cap}."
    )

    try:
        render_schedule_image(
            time_slots=data.get("time_slots", []),
            courses=data.get("courses", []),
            out_path=args.image,
        )
    except Exception as e:
        print(f"Failed to render image: {e}")

if __name__ == "__main__":
    main()
