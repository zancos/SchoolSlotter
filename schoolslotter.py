import json
import os

# Multilingual dictionary
LANGS = {
    "en": {
        "main_menu": "Main Menu",
        "manage_classrooms": "Manage Classrooms",
        "manage_time_slots": "Manage Time Slots",
        "manage_courses": "Manage Courses",
        "generate_schedule": "Generate Schedule",
        "change_language": "Change Language",
        "exit": "Exit",
        "classroom_name": "Classroom Name",
        "priority": "Priority",
        "obligatory_type": "Obligatory Type (at_least_once/at_least_twice/optional)",
        "color_code": "Color Code",
        "add": "Add",
        "edit": "Edit",
        "delete": "Delete",
        "list": "List",
        "back": "Back",
        "start_time": "Start Time",
        "end_time": "End Time",
        "weekday": "Weekday (0=Mon, 1=Tue, ... 6=Sun)",
        "slot_index": "Slot Index (0=first slot of the day)",
        "course_name": "Course Name",
        "course_color": "Course Color",
        "course_time_slots": "Course Time Slots (Example: 0,1;2,2 for Mon 2nd slot & Wed 3rd slot)",
        "invalid_option": "Invalid option.",
        "schedule_generated": "Schedule Generated!",
        "choose_language": "Choose language: 1=en 2=es 3=ca 4=fr 5=it 6=de",
    },
    "es": {
        "main_menu": "Menú Principal",
        "manage_classrooms": "Gestionar Aulas",
        "manage_time_slots": "Gestionar Horarios",
        "manage_courses": "Gestionar Cursos",
        "generate_schedule": "Generar Horario",
        "change_language": "Cambiar Idioma",
        "exit": "Salir",
        "classroom_name": "Nombre del Aula",
        "priority": "Prioridad",
        "obligatory_type": "Tipo de Obligatoriedad (at_least_once/at_least_twice/optional)",
        "color_code": "Código de Color",
        "add": "Agregar",
        "edit": "Editar",
        "delete": "Eliminar",
        "list": "Listar",
        "back": "Volver",
        "start_time": "Hora de Inicio",
        "end_time": "Hora de Fin",
        "weekday": "Día de la Semana (0=Lun, ..., 6=Dom)",
        "slot_index": "Índice de Franja (0=primera franja del día)",
        "course_name": "Nombre del Curso",
        "course_color": "Color del Curso",
        "course_time_slots": "Franjas Horarias del Curso (Ejemplo: 0,1;2,2 para Lunes 2ª y Miércoles 3ª)",
        "invalid_option": "Opción inválida.",
        "schedule_generated": "¡Horario generado!",
        "choose_language": "Elige idioma: 1=en 2=es 3=ca 4=fr 5=it 6=de",
    },
    "ca": {
        "main_menu": "Menú Principal",
        "manage_classrooms": "Gestionar Aules",
        "manage_time_slots": "Gestionar Franges",
        "manage_courses": "Gestionar Cursos",
        "generate_schedule": "Generar Horari",
        "change_language": "Canviar Idioma",
        "exit": "Eixir",
        "classroom_name": "Nom de l'Aula",
        "priority": "Prioritat",
        "obligatory_type": "Tipus d'Obligatorietat (at_least_once/at_least_twice/optional)",
        "color_code": "Codi de Color",
        "add": "Afegir",
        "edit": "Editar",
        "delete": "Eliminar",
        "list": "Llistar",
        "back": "Tornar",
        "start_time": "Hora d'inici",
        "end_time": "Hora de fi",
        "weekday": "Dia de la setmana (0=Dll, ..., 6=Dg)",
        "slot_index": "Índex de Franja (0=primera franja del dia)",
        "course_name": "Nom del Curs",
        "course_color": "Color del Curs",
        "course_time_slots": "Franges del Curs (Exemple: 0,1;2,2 per Dll 2ª i Dmc 3ª)",
        "invalid_option": "Opció invàlida.",
        "schedule_generated": "Horari generat!",
        "choose_language": "Tria idioma: 1=en 2=es 3=ca 4=fr 5=it 6=de",
    },
    "fr": {
        "main_menu": "Menu Principal",
        "manage_classrooms": "Gérer les salles",
        "manage_time_slots": "Gérer les créneaux horaires",
        "manage_courses": "Gérer les cours",
        "generate_schedule": "Générer l'emploi du temps",
        "change_language": "Changer de langue",
        "exit": "Quitter",
        "classroom_name": "Nom de la salle",
        "priority": "Priorité",
        "obligatory_type": "Type d'obligation (at_least_once/at_least_twice/optional)",
        "color_code": "Code couleur",
        "add": "Ajouter",
        "edit": "Modifier",
        "delete": "Supprimer",
        "list": "Lister",
        "back": "Retour",
        "start_time": "Heure de début",
        "end_time": "Heure de fin",
        "weekday": "Jour de la semaine (0=Lun, ..., 6=Dim)",
        "slot_index": "Indice du créneau (0=premier créneau du jour)",
        "course_name": "Nom du cours",
        "course_color": "Couleur du cours",
        "course_time_slots": "Créneaux du cours (Exemple: 0,1;2,2 pour Lun 2ème & Mer 3ème)",
        "invalid_option": "Option invalide.",
        "schedule_generated": "Emploi du temps généré!",
        "choose_language": "Choisissez la langue : 1=en 2=es 3=ca 4=fr 5=it 6=de",
    },
    "it": {
        "main_menu": "Menu Principale",
        "manage_classrooms": "Gestisci Aule",
        "manage_time_slots": "Gestisci Fasce Orarie",
        "manage_courses": "Gestisci Corsi",
        "generate_schedule": "Genera Orario",
        "change_language": "Cambia Lingua",
        "exit": "Esci",
        "classroom_name": "Nome Aula",
        "priority": "Priorità",
        "obligatory_type": "Tipo di Obbligatorietà (at_least_once/at_least_twice/optional)",
        "color_code": "Codice Colore",
        "add": "Aggiungi",
        "edit": "Modifica",
        "delete": "Elimina",
        "list": "Elenca",
        "back": "Indietro",
        "start_time": "Ora di inizio",
        "end_time": "Ora di fine",
        "weekday": "Giorno della settimana (0=Lun, ..., 6=Dom)",
        "slot_index": "Indice Fascia (0=prima fascia del giorno)",
        "course_name": "Nome del Corso",
        "course_color": "Colore del Corso",
        "course_time_slots": "Fasce del Corso (Esempio: 0,1;2,2 per Lun 2° e Mer 3°)",
        "invalid_option": "Opzione non valida.",
        "schedule_generated": "Orario generato!",
        "choose_language": "Scegli lingua: 1=en 2=es 3=ca 4=fr 5=it 6=de",
    },
    "de": {
        "main_menu": "Hauptmenü",
        "manage_classrooms": "Klassenzimmer verwalten",
        "manage_time_slots": "Zeitslots verwalten",
        "manage_courses": "Kurse verwalten",
        "generate_schedule": "Stundenplan generieren",
        "change_language": "Sprache ändern",
        "exit": "Beenden",
        "classroom_name": "Klassenzimmername",
        "priority": "Priorität",
        "obligatory_type": "Verpflichtungstyp (at_least_once/at_least_twice/optional)",
        "color_code": "Farbcode",
        "add": "Hinzufügen",
        "edit": "Bearbeiten",
        "delete": "Löschen",
        "list": "Auflisten",
        "back": "Zurück",
        "start_time": "Anfangszeit",
        "end_time": "Endzeit",
        "weekday": "Wochentag (0=Mo, ..., 6=So)",
        "slot_index": "Slot-Index (0=erster Slot des Tages)",
        "course_name": "Kursname",
        "course_color": "Kursfarbe",
        "course_time_slots": "Kurszeitslots (Bsp: 0,1;2,2 für Mo 2. & Mi 3.)",
        "invalid_option": "Ungültige Option.",
        "schedule_generated": "Stundenplan erstellt!",
        "choose_language": "Sprache wählen: 1=en 2=es 3=ca 4=fr 5=it 6=de",
    }
}

LANG_CODES = ["en", "es", "ca", "fr", "it", "de"]

def tr(key, lang):
    return LANGS[lang][key]

JSON_FILE = "school_data.json"

def load_data():
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, "w") as f:
            f.write(INITIAL_JSON)
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data):
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def input_int(prompt, default=None):
    try:
        return int(input(prompt))
    except:
        return default

### --- MENU SYSTEM ---

def main_menu(lang, data):
    while True:
        print(f"\n=== {tr('main_menu', lang)} ===")
        print(f"1. {tr('manage_classrooms', lang)}")
        print(f"2. {tr('manage_time_slots', lang)}")
        print(f"3. {tr('manage_courses', lang)}")
        print(f"4. {tr('generate_schedule', lang)}")
        print(f"5. {tr('change_language', lang)}")
        print(f"6. {tr('exit', lang)}")
        opt = input("> ").strip()
        if opt == "1":
            classrooms_menu(lang, data)
        elif opt == "2":
            time_slots_menu(lang, data)
        elif opt == "3":
            courses_menu(lang, data)
        elif opt == "4":
            generate_schedule(lang, data)
        elif opt == "5":
            lang = change_language(lang)
        elif opt == "6":
            save_data(data)
            print("Bye!")
            break
        else:
            print(tr("invalid_option", lang))

def classrooms_menu(lang, data):
    while True:
        print(f"\n=== {tr('manage_classrooms', lang)} ===")
        print(f"1. {tr('add', lang)}")
        print(f"2. {tr('list', lang)}")
        print(f"3. {tr('delete', lang)}")
        print(f"4. {tr('back', lang)}")
        opt = input("> ").strip()
        if opt == "1":
            add_classroom(lang, data)
        elif opt == "2":
            list_classrooms(lang, data)
        elif opt == "3":
            del_classroom(lang, data)
        elif opt == "4":
            break
        else:
            print(tr("invalid_option", lang))

def list_classrooms(lang, data):
    print("\n#")
    for i, c in enumerate(data["classrooms"]):
        print(f"{i+1}) {c['name']} | {tr('priority', lang)}: {c['priority']}, {tr('obligatory_type', lang)}: {c['obligatory_type']}, {tr('color_code', lang)}: {c['color_code']}")

def add_classroom(lang, data):
    c = {}
    c["name"] = input(f"{tr('classroom_name', lang)}: ")
    c["priority"] = input_int(f"{tr('priority', lang)}: ", 1)
    c["obligatory_type"] = input(f"{tr('obligatory_type', lang)}: ").strip()
    c["color_code"] = input(f"{tr('color_code', lang)}: ").strip()
    data["classrooms"].append(c)
    save_data(data)

def del_classroom(lang, data):
    list_classrooms(lang, data)
    idx = input_int("Delete number? ", None)
    if idx and 1 <= idx <= len(data["classrooms"]):
        data["classrooms"].pop(idx-1)
        save_data(data)

def time_slots_menu(lang, data):
    while True:
        print(f"\n=== {tr('manage_time_slots', lang)} ===")
        print(f"1. {tr('add', lang)}")
        print(f"2. {tr('list', lang)}")
        print(f"3. {tr('delete', lang)}")
        print(f"4. {tr('back', lang)}")
        opt = input("> ").strip()
        if opt == "1":
            add_time_slot(lang, data)
        elif opt == "2":
            list_time_slots(lang, data)
        elif opt == "3":
            del_time_slot(lang, data)
        elif opt == "4":
            break
        else:
            print(tr("invalid_option", lang))

def list_time_slots(lang, data):
    print("\n#")
    for ts in data["time_slots"]:
        print(f"id:{ts['id']} | {tr('weekday', lang)}:{ts['weekday']} {tr('slot_index', lang)}:{ts['slot_index']} {tr('start_time', lang)}:{ts['start_time']}-{tr('end_time', lang)}:{ts['end_time']}")

def add_time_slot(lang, data):
    ts = {}
    ts["id"] = max([t["id"] for t in data["time_slots"]] + [0]) + 1
    ts["weekday"] = input_int(f"{tr('weekday', lang)}: ", 0)
    ts["slot_index"] = input_int(f"{tr('slot_index', lang)}: ", 0)
    ts["start_time"] = input(f"{tr('start_time', lang)}: ")
    ts["end_time"] = input(f"{tr('end_time', lang)}: ")
    data["time_slots"].append(ts)
    save_data(data)

def del_time_slot(lang, data):
    list_time_slots(lang, data)
    idx = input_int("Delete id? ", None)
    before = len(data["time_slots"])
    data["time_slots"] = [x for x in data["time_slots"] if x["id"] != idx]
    if len(data["time_slots"]) < before:
        save_data(data)

def courses_menu(lang, data):
    while True:
        print(f"\n=== {tr('manage_courses', lang)} ===")
        print(f"1. {tr('add', lang)}")
        print(f"2. {tr('list', lang)}")
        print(f"3. {tr('delete', lang)}")
        print(f"4. {tr('back', lang)}")
        opt = input("> ").strip()
        if opt == "1":
            add_course(lang, data)
        elif opt == "2":
            list_courses(lang, data)
        elif opt == "3":
            del_course(lang, data)
        elif opt == "4":
            break
        else:
            print(tr("invalid_option", lang))

def list_courses(lang, data):
    print("\n#")
    for i, c in enumerate(data["courses"]):
        ts = "; ".join([f"{x[0]},{x[1]}" for x in c["time_slots"]])
        print(f"{i+1}) {c['name']} ({tr('course_color', lang)}: {c['color_code']}) | {tr('course_time_slots', lang)}: {ts}")

def add_course(lang, data):
    c = {}
    c["name"] = input(f"{tr('course_name', lang)}: ")
    c["color_code"] = input(f"{tr('course_color', lang)}: ")
    print(f"{tr('course_time_slots', lang)}")
    print(" - Enter as: 0,1;2,2 (weekday,slot_index ; weekday,slot_index ...)")
    raw = input(" > ")
    time_slots = []
    for part in raw.split(";"):
        if "," in part:
            w, s = part.strip().split(",")
            time_slots.append([int(w), int(s)])
    c["time_slots"] = time_slots
    data["courses"].append(c)
    save_data(data)

def del_course(lang, data):
    list_courses(lang, data)
    idx = input_int("Delete number? ", None)
    if idx and 1 <= idx <= len(data["courses"]):
        data["courses"].pop(idx-1)
        save_data(data)

##### --- SCHEDULER ---

def generate_schedule(lang, data):
    print(f"\n=== {tr('generate_schedule', lang)} ===")
    # Priority order of classrooms
    sorted_classrooms = sorted(data["classrooms"], key=lambda x: x["priority"])
    # Build a lookup map for time_slots
    ts_map = {(t["weekday"], t["slot_index"]): t for t in data["time_slots"]}
    # Output schedule per course
    schedule = {}
    for course in data["courses"]:
        slots = [ts_map.get(tuple(tup)) for tup in course["time_slots"] if tuple(tup) in ts_map]
        usage = {c["name"]:0 for c in data["classrooms"]}
        classroom_schedule = {}
        obligatory_classrooms = [c for c in sorted_classrooms if c["obligatory_type"] in ["at_least_once","at_least_twice"]]
        idx = 0
        needed = {}
        for c in obligatory_classrooms:
            needed[c["name"]] = 2 if c["obligatory_type"]=="at_least_twice" else 1
        for (slot, c_obl) in zip(slots, obligatory_classrooms):
            key = f"{day_name(slot['weekday'],lang)} {slot['start_time']}-{slot['end_time']}"
            classroom_schedule[key] = c_obl["name"]
            usage[c_obl["name"]] += 1
        # Remaining slots: distribute equitably (lowest usage first)
        for slot in slots[len(obligatory_classrooms):]:
            # Classroom with lowest usage, giving priority
            eligible = sorted_classrooms
            eligible = sorted(eligible, key=lambda c: (usage[c["name"]],c["priority"]))
            sel = eligible[0]
            key = f"{day_name(slot['weekday'],lang)} {slot['start_time']}-{slot['end_time']}"
            classroom_schedule[key] = sel["name"]
            usage[sel["name"]] += 1
        schedule[course["name"]] = classroom_schedule
    # Print schedule
    for cname, sched in schedule.items():
        print(f"\n== {cname}")
        for tdesc, aula in sched.items():
            print(f" {tdesc}: {aula}")
    print(f"\n{tr('schedule_generated', lang)}\n")

def day_name(idx, lang):
    names = {
        "en": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        "es": ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"],
        "ca": ["Dll","Dm","Dmc","Dj","Dv","Ds","Dg"],
        "fr": ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"],
        "it": ["Lun","Mar","Mer","Gio","Ven","Sab","Dom"],
        "de": ["Mo","Di","Mi","Do","Fr","Sa","So"],
    }
    return names.get(lang, names["en"])[idx]

def change_language(current):
    print(tr("choose_language", current))
    opt = input("> ").strip()
    idx = 0
    try:
        idx = int(opt)-1
        assert 0 <= idx < len(LANG_CODES)
        return LANG_CODES[idx]
    except Exception:
        print("Invalid. Language unchanged.")
        return current

##### --- DATA (DEFAULT) ---

INITIAL_JSON = '''{
  "classrooms": [
    {"name": "Outdoor Courtyard", "priority": 1, "obligatory_type": "at_least_once", "color_code": "#4CAF50"},
    {"name": "Closed Classroom", "priority": 2, "obligatory_type": "at_least_once", "color_code": "#2196F3"},
    {"name": "Alternative Classroom", "priority": 3, "obligatory_type": "optional", "color_code": "#FF9800"}
  ],
  "time_slots": [
    {"id": 0, "weekday": 0, "slot_index": 0, "start_time": "09:00", "end_time": "10:00"},
    {"id": 1, "weekday": 0, "slot_index": 1, "start_time": "10:00", "end_time": "11:00"},
    {"id": 2, "weekday": 0, "slot_index": 2, "start_time": "11:30", "end_time": "12:30"},
    {"id": 3, "weekday": 0, "slot_index": 3, "start_time": "12:30", "end_time": "13:30"},
    {"id": 4, "weekday": 0, "slot_index": 4, "start_time": "15:00", "end_time": "16:00"},
    {"id": 5, "weekday": 0, "slot_index": 5, "start_time": "16:00", "end_time": "17:00"},
    {"id": 6, "weekday": 0, "slot_index": 6, "start_time": "17:00", "end_time": "18:00"},
    {"id": 7, "weekday": 1, "slot_index": 0, "start_time": "09:00", "end_time": "10:00"},
    {"id": 8, "weekday": 1, "slot_index": 1, "start_time": "10:00", "end_time": "11:00"},
    {"id": 9, "weekday": 1, "slot_index": 2, "start_time": "11:30", "end_time": "12:30"},
    {"id": 10, "weekday": 1, "slot_index": 3, "start_time": "12:30", "end_time": "13:30"},
    {"id": 11, "weekday": 1, "slot_index": 4, "start_time": "15:00", "end_time": "16:00"},
    {"id": 12, "weekday": 1, "slot_index": 5, "start_time": "16:00", "end_time": "17:00"},
    {"id": 13, "weekday": 1, "slot_index": 6, "start_time": "17:00", "end_time": "18:00"},
    {"id": 14, "weekday": 2, "slot_index": 0, "start_time": "09:00", "end_time": "10:00"},
    {"id": 15, "weekday": 2, "slot_index": 1, "start_time": "10:00", "end_time": "11:00"},
    {"id": 16, "weekday": 2, "slot_index": 2, "start_time": "11:30", "end_time": "12:30"},
    {"id": 17, "weekday": 2, "slot_index": 3, "start_time": "12:30", "end_time": "13:30"},
    {"id": 18, "weekday": 2, "slot_index": 4, "start_time": "15:00", "end_time": "16:00"},
    {"id": 19, "weekday": 2, "slot_index": 5, "start_time": "16:00", "end_time": "17:00"},
    {"id": 20, "weekday": 2, "slot_index": 6, "start_time": "17:00", "end_time": "18:00"},
    {"id": 21, "weekday": 3, "slot_index": 0, "start_time": "09:00", "end_time": "10:00"},
    {"id": 22, "weekday": 3, "slot_index": 1, "start_time": "10:00", "end_time": "11:00"},
    {"id": 23, "weekday": 3, "slot_index": 2, "start_time": "11:30", "end_time": "12:30"},
    {"id": 24, "weekday": 3, "slot_index": 3, "start_time": "12:30", "end_time": "13:30"},
    {"id": 25, "weekday": 3, "slot_index": 4, "start_time": "15:00", "end_time": "16:00"},
    {"id": 26, "weekday": 3, "slot_index": 5, "start_time": "16:00", "end_time": "17:00"},
    {"id": 27, "weekday": 3, "slot_index": 6, "start_time": "17:00", "end_time": "18:00"},
    {"id": 28, "weekday": 4, "slot_index": 0, "start_time": "09:00", "end_time": "10:00"},
    {"id": 29, "weekday": 4, "slot_index": 1, "start_time": "10:00", "end_time": "11:00"},
    {"id": 30, "weekday": 4, "slot_index": 2, "start_time": "11:30", "end_time": "12:30"},
    {"id": 31, "weekday": 4, "slot_index": 3, "start_time": "12:30", "end_time": "13:30"},
    {"id": 32, "weekday": 4, "slot_index": 4, "start_time": "15:00", "end_time": "16:00"},
    {"id": 33, "weekday": 4, "slot_index": 5, "start_time": "16:00", "end_time": "17:00"},
    {"id": 34, "weekday": 4, "slot_index": 6, "start_time": "17:00", "end_time": "18:00"}
  ],
  "courses": [
    {"name": "1st Grade Primary", "color_code": "#F44336", "time_slots": [[0,0],[2,1]]},
    {"name": "2nd Grade Primary", "color_code": "#E91E63", "time_slots": [[1,2],[3,3]]},
    {"name": "3rd Grade Primary", "color_code": "#9C27B0", "time_slots": [[0,1],[4,3]]},
    {"name": "4th Grade Primary", "color_code": "#673AB7", "time_slots": [[1,0],[2,0]]},
    {"name": "5th Grade Primary", "color_code": "#3F51B5", "time_slots": [[0,2],[3,1]]},
    {"name": "6th Grade Primary", "color_code": "#2196F3", "time_slots": [[2,2],[4,0]]},
    {"name": "1st Grade Secondary", "color_code": "#03A9F4", "time_slots": [[1,1],[3,0],[4,2]]},
    {"name": "2nd Grade Secondary", "color_code": "#00BCD4", "time_slots": [[2,1],[3,2],[4,1]]},
    {"name": "3rd Grade Secondary", "color_code": "#009688", "time_slots": [[0,3],[2,3],[4,3]]},
    {"name": "4th Grade Secondary", "color_code": "#4CAF50", "time_slots": [[0,4],[1,3],[2,4]]},
    {"name": "1st Grade Bachillerato", "color_code": "#8BC34A", "time_slots": [[0,5],[2,5],[4,4]]},
    {"name": "2nd Grade Bachillerato", "color_code": "#CDDC39", "time_slots": [[1,4],[3,4],[4,5]]},
    {"name": "Advanced Course A", "color_code": "#FFEB3B", "time_slots": [[0,6],[2,6],[4,6]]},
    {"name": "Advanced Course B", "color_code": "#FFC107", "time_slots": [[1,5],[3,5],[4,6]]},
    {"name": "Advanced Course C", "color_code": "#FF9800", "time_slots": [[1,6],[2,0],[3,6]]}
  ]
}'''

#### ---- MAIN ----

if __name__ == "__main__":
    data = load_data()
    lang = "en"
    main_menu(lang, data)
