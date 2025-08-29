#!/usr/bin/env python3
# schoolslotter_exact.py
# Exact classroom assignment using CP-SAT (OR-Tools), with feasibility pre-checks, multi-core solving,
# and optional PNG+PDF report generation (requires pillow + reportlab).

import argparse
import json
import multiprocessing as mp
from collections import defaultdict, Counter
import os

# Optional deps for PNG/PDF
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors as rl_colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# OR-Tools
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except Exception:
    ORTOOLS_AVAILABLE = False

WEEKDAY_NAMES = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}

# ---------------- Feasibility Checks ----------------

def necessary_feasibility_checks(data, max_share=1):
    courses = data.get('courses', [])
    classrooms = data.get('classrooms', [])
    time_slots = data.get('time_slots', [])

    pair_set = {(ts['weekday'], ts['slot_index']) for ts in time_slots}
    num_pairs = len(pair_set)

    rooms = [r['name'] for r in classrooms]
    mandatory_rooms = [r['name'] for r in classrooms if r.get('obligatory_type') == 'at_least_once']

    per_slot_sessions = Counter()
    course_sessions = {}
    for ci, c in enumerate(courses):
        ts_list = c.get('time_slots', [])
        course_sessions[c.get('name', f'Course {ci}')] = len(ts_list)
        for d, s in ts_list:
            per_slot_sessions[(d, s)] += 1

    checks = {
        'courses_have_enough_sessions': [],
        'slot_capacity_ok': [],
        'room_weekly_capacity_ok': [],
        'summary': {}
    }

    m = len(mandatory_rooms)
    ok_sessions = True
    for cname, n in course_sessions.items():
        ok = (n >= m)
        checks['courses_have_enough_sessions'].append({'course': cname, 'sessions': n, 'required_min': m, 'ok': ok})
        if not ok:
            ok_sessions = False

    R = len(rooms)
    ok_slots = True
    for (d, s), cnt in per_slot_sessions.items():
        cap = R * max_share
        ok = (cnt <= cap)
        checks['slot_capacity_ok'].append({'weekday': d, 'slot_index': s, 'sessions': cnt, 'capacity': cap, 'ok': ok})
        if not ok:
            ok_slots = False

    ok_room_week = True
    C = len(courses)
    for rn in mandatory_rooms:
        cap = num_pairs * max_share
        ok = (C <= cap)
        checks['room_weekly_capacity_ok'].append({'room': rn, 'courses': C, 'room_weekly_capacity': cap, 'ok': ok})
        if not ok:
            ok_room_week = False

    overall = ok_sessions and ok_slots and ok_room_week
    checks['summary'] = {
        'num_courses': C,
        'num_rooms': R,
        'num_mandatory_rooms': m,
        'num_weekly_pairs': num_pairs,
        'overall_necessary_ok': overall
    }
    return overall, checks

# ---------------- Exact Solver ----------------

def solve_exact(data, threads=None, max_share=1, time_limit=None):
    if not ORTOOLS_AVAILABLE:
        raise RuntimeError("OR-Tools no está instalado. pip install ortools")

    classrooms = data.get('classrooms', [])
    courses = data.get('courses', [])
    time_slots = data.get('time_slots', [])

    room_list = [r['name'] for r in classrooms]
    room_index = {rn: i for i, rn in enumerate(room_list)}
    mandatory_rooms = [r['name'] for r in classrooms if r.get('obligatory_type') == 'at_least_once']

    sessions = []  # (ci, cname, ki, (d,s))
    for ci, c in enumerate(courses):
        cname = c.get('name', f'Course {ci}')
        for ki, (d, s) in enumerate(c.get('time_slots', [])):
            sessions.append((ci, cname, ki, (d, s)))

    per_slot = defaultdict(list)
    for (ci, cname, ki, (d, s)) in sessions:
        per_slot[(d, s)].append((ci, ki))

    # Priority weights
    max_pr = max((r.get('priority', 999) for r in classrooms), default=1)
    room_weight = {}
    for r in classrooms:
        pr = r.get('priority', max_pr)
        room_weight[r['name']] = (max_pr + 1 - pr) * 1000

    model = cp_model.CpModel()

    X = {}
    for (ci, cname, ki, (d, s)) in sessions:
        for rn in room_list:
            X[(ci, ki, rn)] = model.NewBoolVar(f"x_c{ci}_k{ki}_r{room_index[rn]}")

    # Each session to exactly one room
    for (ci, cname, ki, (d, s)) in sessions:
        model.Add(sum(X[(ci, ki, rn)] for rn in room_list) == 1)

    # Per-slot room capacity
    for (d, s), ck_list in per_slot.items():
        for rn in room_list:
            model.Add(sum(X[(ci, ki, rn)] for (ci, ki) in ck_list) <= max_share)

    # Obligations per course and mandatory room
    per_course_sessions = defaultdict(list)
    for (ci, cname, ki, (d, s)) in sessions:
        per_course_sessions[ci].append((ci, ki))
    for ci in per_course_sessions.keys():
        for rn in mandatory_rooms:
            model.Add(sum(X[(ci, ki, rn)] for (_, ki) in per_course_sessions[ci]) >= 1)

    # Objective: maximize priority preference
    obj_terms = []
    for (ci, cname, ki, (d, s)) in sessions:
        for rn in room_list:
            w = room_weight.get(rn, 0)
            if w:
                obj_terms.append(w * X[(ci, ki, rn)])
    if obj_terms:
        model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    threads = (threads if threads and threads>0 else mp.cpu_count())
    solver.parameters.num_search_workers = threads
    if time_limit:
        solver.parameters.max_time_in_seconds = float(time_limit)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, {'status': int(status), 'status_name': solver.StatusName(status)}

    assigned = defaultdict(list)
    for (ci, cname, ki, (d, s)) in sessions:
        for rn in room_list:
            if solver.Value(X[(ci, ki, rn)]) == 1:
                assigned[ci].append([d, s, rn])
                break

    courses_out = []
    for ci, c in enumerate(courses):
        courses_out.append({
            'name': c.get('name'),
            'color_code': c.get('color_code'),
            'time_slots': c.get('time_slots', []),
            'classroom_slots': assigned.get(ci, [])
        })

    usage = Counter()
    daily_usage = defaultdict(Counter)
    per_course_usage = {}
    for c in courses_out:
        cu = Counter()
        for d, s, rn in c['classroom_slots']:
            usage[rn] += 1
            daily_usage[d][rn] += 1
            cu[rn] += 1
        per_course_usage[c['name']] = dict(cu)

    violations = []
    for ci, c in enumerate(courses):
        cname = c.get('name')
        assigned_rooms = [rn for d, s, rn in courses_out[ci]['classroom_slots']]
        for rn in mandatory_rooms:
            if rn not in assigned_rooms:
                violations.append({'course': cname, 'missing_room': rn})

    result = {
        'classrooms': classrooms,
        'time_slots': time_slots,
        'courses': courses_out,
        'stats': {
            'classroom_usage': dict(usage),
            'daily_usage': {str(d): dict(cnt) for d, cnt in daily_usage.items()},
            'per_course_usage': per_course_usage,
            'violations': violations
        },
        'meta': {
            'threads': threads,
            'solver_status': solver.StatusName(status),
            'objective_value': solver.ObjectiveValue() if obj_terms else None
        }
    }
    return result, None

# ---------------- Rendering ----------------

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    try:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (204, 204, 204)


def build_axes(time_slots):
    weekdays = sorted({ts['weekday'] for ts in time_slots})
    slot_times = {}
    for ts in time_slots:
        si = ts['slot_index']
        if si not in slot_times:
            slot_times[si] = f"{ts['start_time']}-{ts['end_time']}"
    slot_indexes = sorted(slot_times.keys())
    return weekdays, slot_indexes, slot_times


def sort_classrooms(classrooms):
    return sorted(classrooms, key=lambda r: (r.get('priority', 9999), r.get('name', '')))


def render_schedule_image(payload, out_png='room_schedule_exact.png'):
    if not PIL_AVAILABLE:
        print("[Aviso] Pillow no está instalado; no se genera PNG. pip install pillow")
        return None

    time_slots = payload.get('time_slots', [])
    classrooms = payload.get('classrooms', [])
    courses = payload.get('courses', [])

    weekdays, slot_indexes, slot_times = build_axes(time_slots)

    room_color = {r['name']: r.get('color_code', '#CCCCCC') for r in classrooms}
    cell = defaultdict(list)
    for c in courses:
        cname = c.get('name', '')
        for d, s, rn in c.get('classroom_slots', []):
            cell[(d, s)].append((rn, room_color.get(rn, '#DDDDDD'), cname))
    pr_order = {r['name']: idx for idx, r in enumerate(sort_classrooms(classrooms))}

    label_w, header_h = 220, 56
    cell_w, cell_h = 320, 140
    swatch_h, swatch_w = 14, 14
    pad_x, pad_y = 8, 8

    cols, rows = len(weekdays), len(slot_indexes)
    width, height = label_w + cols * cell_w, header_h + rows * cell_h

    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 16)
        small = ImageFont.truetype('arial.ttf', 13)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    draw.rectangle([0, 0, width-1, header_h-1], fill=(235,242,250))
    draw.rectangle([0, 0, label_w-1, height-1], fill=(246,246,246))

    for ci, d in enumerate(weekdays):
        x0 = label_w + ci * cell_w
        draw.rectangle([x0, 0, x0 + cell_w, header_h], outline=(180,180,180))
        title = WEEKDAY_NAMES.get(d, f'Día {d}')
        try:
            tw = draw.textlength(title, font=font)
        except Exception:
            tw = len(title) * 8
        draw.text((x0 + (cell_w - tw)/2, (header_h - 22)/2), title, fill=(0,0,0), font=font)

    for ri, s in enumerate(slot_indexes):
        y0 = header_h + ri * cell_h
        label = slot_times.get(s, f'Tramo {s}')
        draw.rectangle([0, y0, label_w, y0 + cell_h], outline=(180,180,180))
        draw.text((10, y0 + 10), label, fill=(0,0,0), font=font)

    for ri, s in enumerate(slot_indexes):
        for ci, d in enumerate(weekdays):
            x0 = label_w + ci * cell_w
            y0 = header_h + ri * cell_h
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=(200,200,200))
            items = sorted(cell.get((d, s), []), key=lambda t: pr_order.get(t[0], 9999))
            cur_y = y0 + pad_y
            for rn, hexcol, cname in items:
                rgb = hex_to_rgb(hexcol)
                sw_x0, sw_y0 = x0 + pad_x, cur_y
                sw_x1, sw_y1 = sw_x0 + swatch_w, sw_y0 + swatch_h
                draw.rectangle([sw_x0, sw_y0, sw_x1, sw_y1], fill=rgb, outline=(120,120,120))
                txt = f"{cname} — {rn}" if rn else f"{cname} — SIN AULA"
                draw.text((sw_x1 + 6, sw_y0 - 1), txt, fill=(0,0,0), font=small)
                cur_y += swatch_h + 6
                if cur_y > y0 + cell_h - swatch_h:
                    break

    img.save(out_png)
    return out_png


def render_pdf_report(payload, schedule_png, out_pdf='schoolslotter_exact_report.pdf'):
    if not REPORTLAB_AVAILABLE:
        print("[Aviso] reportlab no está instalado; no se genera PDF. pip install reportlab")
        return None

    doc = SimpleDocTemplate(out_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph('<b>Horario (Asignación Exacta)</b>', styles['Title']))
    story.append(Spacer(1, 8))

    rooms_sorted = sort_classrooms(payload.get('classrooms', []))
    legend_rows = [["Aula", "Prioridad", "Obligatorio", "Color"]]
    for r in rooms_sorted:
        legend_rows.append([r.get('name'), str(r.get('priority')), r.get('obligatory_type'), r.get('color_code', '')])
    legend_tbl = Table(legend_rows, hAlign='LEFT', colWidths=[180, 60, 100, 120])
    ts = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), rl_colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, rl_colors.grey),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
    ])
    # Paint color swatch in color column
    def _hex_to_rgb_safe(h):
        try:
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (0,2,4))
        except Exception:
            return (204,204,204)
    for i, r in enumerate(rooms_sorted, start=1):
        hexcol = r.get('color_code', '#CCCCCC')
        try:
            ts.add('BACKGROUND', (3, i), (3, i), rl_colors.HexColor(hexcol))
            rgb = _hex_to_rgb_safe(hexcol)
            luminance = (0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]) / 255.0
            txt_color = rl_colors.black if luminance > 0.6 else rl_colors.white
            ts.add('TEXTCOLOR', (3, i), (3, i), txt_color)
        except Exception:
            pass
    legend_tbl.setStyle(ts)

    story.append(legend_tbl)
    story.append(Spacer(1, 10))

    if schedule_png and os.path.exists(schedule_png):
        story.append(Paragraph('<b>Cuadrícula Semanal</b>', styles['Heading2']))
        story.append(RLImage(schedule_png, width=520, height=520))
        story.append(Spacer(1, 10))

    # Tables with usage and violations
    usage = payload.get('stats', {}).get('classroom_usage', {})
    if usage:
        rows = [["Aula", "Sesiones"]] + [[k, str(v)] for k, v in sorted(usage.items())]
        tbl = Table(rows, hAlign='LEFT')
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), rl_colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, rl_colors.grey),
        ]))
        story.append(Paragraph('<b>Uso por Aula</b>', styles['Heading2']))
        story.append(tbl)
        story.append(Spacer(1, 8))

    daily = payload.get('stats', {}).get('daily_usage', {})
    if daily:
        all_rooms = sorted({room for dct in daily.values() for room in dct.keys()})
        header = ["Día"] + all_rooms
        rows = [header]
        for d, dct in sorted(daily.items(), key=lambda x: int(x[0])):
            rows.append([WEEKDAY_NAMES.get(int(d), d)] + [str(dct.get(r, 0)) for r in all_rooms])
        tbl = Table(rows, hAlign='LEFT')
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), rl_colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, rl_colors.grey),
            ('ALIGN', (1,1), (-1,-1), 'CENTER'),
        ]))
        story.append(Paragraph('<b>Uso Diario (apilado por aula)</b>', styles['Heading2']))
        story.append(tbl)
        story.append(Spacer(1, 8))

    per_course = payload.get('stats', {}).get('per_course_usage', {})
    if per_course:
        story.append(Paragraph('<b>Uso por Curso</b>', styles['Heading2']))
        for cname, dct in sorted(per_course.items()):
            rows = [["Aula", "Sesiones"]] + [[k, str(v)] for k, v in sorted(dct.items())]
            tbl = Table(rows, hAlign='LEFT')
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), rl_colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.25, rl_colors.grey),
            ]))
            story.append(Paragraph(f'<b>{cname}</b>', styles['Heading4']))
            story.append(tbl)
            story.append(Spacer(1, 6))

    viol = payload.get('stats', {}).get('violations', [])
    story.append(Paragraph('<b>Incumplimientos (at_least_once)</b>', styles['Heading2']))
    if viol:
        rows = [["Curso", "Aula requerida"]] + [[v['course'], v['missing_room']] for v in viol]
        tbl = Table(rows, hAlign='LEFT')
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), rl_colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, rl_colors.grey),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph('Todos los cursos cumplen las aulas obligatorias al menos una vez.', styles['Normal']))

    doc.build(story)
    return out_pdf

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description='Asignación exacta de aulas con CP-SAT (multi-core), comprobación de factibilidad y PDF.')
    ap.add_argument('--json', default='school_data.json', help='JSON de entrada con classrooms, time_slots, courses.')
    ap.add_argument('--out_json', default='schoolslotter_output_exact.json', help='JSON de salida.')
    ap.add_argument('--threads', type=int, default=0, help='Hilos (0=todos los cores).')
    ap.add_argument('--max_share', type=int, default=1, help='Capacidad por aula y tramo (1=no compartir).')
    ap.add_argument('--time_limit', type=float, default=None, help='Límite de tiempo del solver (segundos).')
    ap.add_argument('--pdf', default='schoolslotter_exact_report.pdf', help='Ruta del PDF (opcional).')
    ap.add_argument('--image', default='room_schedule_exact.png', help='Ruta del PNG (opcional).')
    args = ap.parse_args()

    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ok, report = necessary_feasibility_checks(data, max_share=args.max_share)
    if not ok:
        print('[Pre-check] Condiciones necesarias NO satisfechas (puede no existir solución).')
    else:
        print('[Pre-check] Condiciones necesarias satisfechas; se intenta resolver exactamente.')

    if not ORTOOLS_AVAILABLE:
        print('ERROR: OR-Tools no está instalado. pip install ortools')
        return

    threads = args.threads if args.threads and args.threads>0 else mp.cpu_count()
    result, err = solve_exact(data, threads=threads, max_share=args.max_share, time_limit=args.time_limit)
    if err is not None:
        print('No hay solución factible (CP-SAT):', err)
        return

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print('Escrito', args.out_json)

    # Render PNG + PDF si hay dependencias
    png = render_schedule_image(result, out_png=args.image)
    if png:
        print('Escrito', png)
    pdf = render_pdf_report(result, schedule_png=png, out_pdf=args.pdf)
    if pdf:
        print('Escrito', pdf)

if __name__ == '__main__':
    main()
