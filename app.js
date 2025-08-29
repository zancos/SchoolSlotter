const { createApp } = Vue;

createApp({
    data() {
        return {
            // Data structure
            timeSlots: [],
            classrooms: [
                {
                    name: "Aula Principal",
                    obligatory_type: "at_least_once",
                    color_code: "#2563eb"
                },
                {
                    name: "Aula Secundaria", 
                    obligatory_type: "at_least_once",
                    color_code: "#059669"
                },
                {
                    name: "Aula Opcional",
                    obligatory_type: "optional",
                    color_code: "#64748b"
                }
            ],
            courses: [],

            // Assignment results
            assignedSchedule: null,

            // UI state
            activeTab: 'timeslots',
            showLoadDialog: false,
            showAddSession: false,
            selectedCourseIndex: -1,
            newSession: { weekday: '', slot_index: '' },
            generating: false,
            statusMessage: '',
            statusType: 'info',

            // Feasibility
            feasibilityStatus: 'unknown',
            feasibilityMessage: 'Configura cursos y aulas para verificar factibilidad',
            isFeasible: false,

            // Constants
            weekdays: ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'],
            nextSlotId: 0,
            nextSlotIndex: 0,

            tabs: [
                { id: 'timeslots', name: 'Horarios', icon: '⏰' },
                { id: 'classrooms', name: 'Aulas', icon: '🏛️' },
                { id: 'courses', name: 'Cursos', icon: '📚' }
            ]
        }
    },

    computed: {
        hasResults() {
            return this.assignedSchedule !== null;
        },

        uniqueTimeSlots() {
            const slots = {};
            this.timeSlots.forEach(slot => {
                const key = `${slot.slot_index}`;
                if (!slots[key]) {
                    slots[key] = {
                        slot_index: slot.slot_index,
                        start_time: slot.start_time,
                        end_time: slot.end_time,
                        key: key
                    };
                }
            });
            return Object.values(slots).sort((a, b) => a.slot_index - b.slot_index);
        }
    },

    watch: {
        classrooms: {
            handler() {
                this.updatePriorities();
                this.checkFeasibility();
            },
            deep: true
        },
        courses: {
            handler() {
                this.checkFeasibility();
            },
            deep: true
        },
        timeSlots: {
            handler() {
                this.checkFeasibility();
            },
            deep: true
        }
    },

    methods: {
        // Feasibility checking
        checkFeasibility() {
            this.$nextTick(() => {
                if (this.courses.length === 0 || this.classrooms.length === 0 || this.timeSlots.length === 0) {
                    this.feasibilityStatus = 'unknown';
                    this.feasibilityMessage = 'Configura cursos, aulas y tramos horarios';
                    this.isFeasible = false;
                    return;
                }

                const result = this.performFeasibilityCheck();
                this.feasibilityStatus = result.feasible ? 'feasible' : 'infeasible';
                this.feasibilityMessage = result.message;
                this.isFeasible = result.feasible;
            });
        },

        performFeasibilityCheck() {
            const mandatoryRooms = this.classrooms.filter(r => r.obligatory_type === 'at_least_once');
            const totalRooms = this.classrooms.length;

            const uniquePairs = new Set();
            this.timeSlots.forEach(slot => {
                uniquePairs.add(`${slot.weekday}-${slot.slot_index}`);
            });
            const totalSlots = uniquePairs.size;

            // Check 1: Each course must have at least as many sessions as mandatory rooms
            for (const course of this.courses) {
                if (course.time_slots.length < mandatoryRooms.length) {
                    return {
                        feasible: false,
                        message: `${course.name} tiene ${course.time_slots.length} sesiones pero necesita al menos ${mandatoryRooms.length} (una por aula obligatoria)`
                    };
                }
            }

            // Check 2: Per-slot capacity
            const slotOccupancy = {};
            this.courses.forEach(course => {
                course.time_slots.forEach(([weekday, slotIndex]) => {
                    const key = `${weekday}-${slotIndex}`;
                    slotOccupancy[key] = (slotOccupancy[key] || 0) + 1;
                });
            });

            for (const [slot, count] of Object.entries(slotOccupancy)) {
                if (count > totalRooms) {
                    return {
                        feasible: false,
                        message: `El tramo ${slot} tiene ${count} cursos pero solo hay ${totalRooms} aulas disponibles`
                    };
                }
            }

            // Check 3: Weekly capacity per mandatory room
            const totalCourses = this.courses.length;
            for (const room of mandatoryRooms) {
                const weeklyCapacity = totalSlots;
                if (totalCourses > weeklyCapacity) {
                    return {
                        feasible: false,
                        message: `${room.name} (obligatoria) debe alojar ${totalCourses} cursos pero solo tiene ${weeklyCapacity} tramos semanales`
                    };
                }
            }

            return {
                feasible: true,
                message: `✅ Configuración viable: ${this.courses.length} cursos, ${mandatoryRooms.length} aulas obligatorias, ${totalSlots} tramos`
            };
        },

        // Priority management
        updatePriorities() {
            this.classrooms.forEach((room, index) => {
                room.priority = index + 1;
            });
        },

        // File handling
        loadJSONFile(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);
                    this.loadData(data);
                    this.showStatus('Archivo cargado correctamente', 'success');
                    this.showLoadDialog = false;
                } catch (error) {
                    this.showStatus('Error al cargar el archivo: ' + error.message, 'error');
                }
            };
            reader.readAsText(file);
        },

        loadData(data) {
            if (data.time_slots) {
                this.timeSlots = data.time_slots.map(slot => ({
                    ...slot,
                    id: this.nextSlotId++
                }));
                this.nextSlotIndex = Math.max(...this.timeSlots.map(s => s.slot_index), -1) + 1;
            }
            if (data.classrooms) {
                this.classrooms = data.classrooms;
                this.updatePriorities();
            }
            if (data.courses) {
                this.courses = data.courses;
            }
            this.checkFeasibility();
        },

        downloadJSON() {
            if (!this.assignedSchedule) return;

            const blob = new Blob([JSON.stringify(this.assignedSchedule, null, 2)], {
                type: 'application/json'
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'schedule_output.json';
            a.click();
            URL.revokeObjectURL(url);
        },

        // Time slot management
        addTimeSlot() {
            this.timeSlots.push({
                id: this.nextSlotId++,
                weekday: 0,
                slot_index: this.nextSlotIndex++,
                start_time: '09:00',
                end_time: '10:00'
            });
        },

        removeTimeSlot(index) {
            this.timeSlots.splice(index, 1);
        },

        // Classroom management
        addClassroom() {
            this.classrooms.push({
                name: 'Nueva Aula',
                obligatory_type: 'optional',
                color_code: this.getRandomColor()
            });
            this.updatePriorities();
        },

        removeClassroom(index) {
            this.classrooms.splice(index, 1);
            this.updatePriorities();
        },

        // Course management
        addCourse() {
            this.courses.push({
                name: 'Nuevo Curso',
                color_code: this.getRandomColor(),
                time_slots: []
            });
        },

        removeCourse(index) {
            this.courses.splice(index, 1);
        },

        // Session management
        showSessionDialog(courseIndex) {
            this.selectedCourseIndex = courseIndex;
            this.newSession = { weekday: '', slot_index: '' };
            this.showAddSession = true;
        },

        addSession() {
            if (this.newSession.weekday === '' || this.newSession.slot_index === '') return;

            const course = this.courses[this.selectedCourseIndex];
            const session = [parseInt(this.newSession.weekday), parseInt(this.newSession.slot_index)];

            const exists = course.time_slots.some(slot => 
                slot[0] === session[0] && slot[1] === session[1]
            );

            if (!exists) {
                course.time_slots.push(session);
            }

            this.showAddSession = false;
        },

        removeSession(courseIndex, sessionIndex) {
            this.courses[courseIndex].time_slots.splice(sessionIndex, 1);
        },

        getAvailableSlots(weekday) {
            if (weekday === '') return [];
            return this.timeSlots.filter(slot => slot.weekday === parseInt(weekday));
        },

        // Grid helpers
        hasDay(dayIndex) {
            return this.timeSlots.some(slot => slot.weekday === dayIndex);
        },

        getCellContent(weekday, slotIndex) {
            const content = [];

            if (this.assignedSchedule && this.assignedSchedule.courses) {
                this.assignedSchedule.courses.forEach(course => {
                    course.classroom_slots.forEach(slot => {
                        if (slot[0] === weekday && slot[1] === slotIndex) {
                            content.push({
                                course: course.name,
                                color: course.color_code,
                                room: slot[2]
                            });
                        }
                    });
                });
            } else {
                // Show original course assignments before solving
                this.courses.forEach(course => {
                    const hasSession = course.time_slots.some(slot => 
                        slot[0] === weekday && slot[1] === slotIndex
                    );
                    if (hasSession) {
                        content.push({
                            course: course.name,
                            color: course.color_code,
                            room: null
                        });
                    }
                });
            }

            return content;
        },

        formatSession(session) {
            const day = this.weekdays[session[0]] || 'Día ' + session[0];
            const timeSlot = this.timeSlots.find(slot => 
                slot.weekday === session[0] && slot.slot_index === session[1]
            );
            const time = timeSlot ? `${timeSlot.start_time}-${timeSlot.end_time}` : `Tramo ${session[1]}`;
            return `${day} ${time}`;
        },

        formatSessionShort(session) {
            const day = this.weekdays[session[0]] || 'D' + session[0];
            const timeSlot = this.timeSlots.find(slot => 
                slot.weekday === session[0] && slot.slot_index === session[1]
            );
            const time = timeSlot ? timeSlot.start_time : `T${session[1]}`;
            return `${day} ${time}`;
        },

        // IMPROVED EXACT SOLVER - Based on working Python algorithm
        async generateSchedule() {
            if (!this.isFeasible) {
                this.showStatus('La configuración actual no es factible. Revisa las restricciones.', 'error');
                return;
            }

            this.generating = true;
            this.showStatus('Generando horario exacto...', 'info');

            try {
                await new Promise(resolve => setTimeout(resolve, 500));

                const result = this.solveExactImproved();

                if (result.success) {
                    this.assignedSchedule = {
                        classrooms: this.classrooms,
                        time_slots: this.timeSlots,
                        courses: result.courses,
                        stats: result.stats,
                        meta: {
                            algorithm: 'Improved Exact Solver (Python-based)',
                            generated_at: new Date().toISOString()
                        }
                    };

                    const violations = result.stats.violations.length;
                    if (violations === 0) {
                        this.showStatus('¡Horario generado con éxito! Todas las restricciones cumplidas.', 'success');
                    } else {
                        this.showStatus(`Error interno: ${violations} violación(es) detectadas en solver mejorado.`, 'error');
                        console.error('Violations:', result.stats.violations);
                    }
                } else {
                    this.showStatus('Error en el solver: ' + result.error, 'error');
                }
            } catch (error) {
                this.showStatus('Error al generar horario: ' + error.message, 'error');
                console.error('Solver error:', error);
            } finally {
                this.generating = false;
            }
        },

        // Improved solver based on the working Python algorithm
        solveExactImproved() {
            console.log('Starting improved solver...');

            const obligatoryRooms = this.classrooms.filter(r => r.obligatory_type === 'at_least_once');
            const optionalRooms = this.classrooms.filter(r => r.obligatory_type !== 'at_least_once');
            const allRooms = [...this.classrooms].sort((a, b) => a.priority - b.priority);

            console.log('Obligatory rooms:', obligatoryRooms.map(r => r.name));
            console.log('All rooms (by priority):', allRooms.map(r => `${r.name}(${r.priority})`));

            // Initialize assignment: courseIndex -> sessionIndex -> roomName
            const assignment = {};
            const occupancy = {}; // (weekday,slot_index,room_name) -> count

            this.courses.forEach((course, ci) => {
                assignment[ci] = new Array(course.time_slots.length).fill(null);
            });

            // Helper function
            const roomAvailable = (d, s, roomName) => {
                const key = `${d},${s},${roomName}`;
                return (occupancy[key] || 0) === 0;
            };

            const setRoomOccupancy = (d, s, roomName, count) => {
                const key = `${d},${s},${roomName}`;
                occupancy[key] = count;
            };

            // PASS A: Satisfy at_least_once for each course and each mandatory room
            console.log('PASS A: Satisfying at_least_once constraints...');

            for (let ci = 0; ci < this.courses.length; ci++) {
                const course = this.courses[ci];
                console.log(`Processing course ${course.name} with ${course.time_slots.length} sessions`);

                // Create shuffled list of sessions for randomness
                const sessions = course.time_slots.map((slot, si) => ({ si, d: slot[0], s: slot[1] }));
                this.shuffleArray(sessions);

                for (const room of obligatoryRooms) {
                    const roomName = room.name;

                    // Check if already fulfilled
                    const alreadyFulfilled = assignment[ci].some(assignedRoom => assignedRoom === roomName);
                    if (alreadyFulfilled) {
                        console.log(`  Course ${course.name} already uses room ${roomName}`);
                        continue;
                    }

                    let placed = false;

                    // Try to place one session in this mandatory room
                    for (const session of sessions) {
                        const { si, d, s } = session;

                        if (assignment[ci][si] !== null) {
                            continue; // Already assigned
                        }

                        if (roomAvailable(d, s, roomName)) {
                            assignment[ci][si] = roomName;
                            setRoomOccupancy(d, s, roomName, 1);
                            console.log(`  Assigned ${course.name} session ${si} to ${roomName} at (${d},${s})`);
                            placed = true;
                            break;
                        }
                    }

                    // If couldn't place, try to reassign from optional room
                    if (!placed) {
                        console.log(`  Trying to reassign for ${course.name} -> ${roomName}`);
                        for (const session of sessions) {
                            const { si, d, s } = session;
                            const currentRoom = assignment[ci][si];

                            if (currentRoom === null) continue;

                            // Only reassign if current room is optional or lower priority
                            const currentIsOptional = optionalRooms.some(r => r.name === currentRoom);
                            const currentPriority = allRooms.find(r => r.name === currentRoom)?.priority || 999;
                            const targetPriority = room.priority;

                            if ((currentIsOptional || currentPriority > targetPriority) && roomAvailable(d, s, roomName)) {
                                // Release current room
                                setRoomOccupancy(d, s, currentRoom, 0);
                                // Assign to mandatory room
                                assignment[ci][si] = roomName;
                                setRoomOccupancy(d, s, roomName, 1);
                                console.log(`  Reassigned ${course.name} session ${si} from ${currentRoom} to ${roomName}`);
                                placed = true;
                                break;
                            }
                        }
                    }

                    if (!placed) {
                        console.warn(`  Failed to assign ${course.name} to mandatory room ${roomName}`);
                    }
                }
            }

            // PASS B: Fill remaining sessions by priority
            console.log('PASS B: Filling remaining sessions by priority...');

            for (let ci = 0; ci < this.courses.length; ci++) {
                const course = this.courses[ci];

                for (let si = 0; si < course.time_slots.length; si++) {
                    if (assignment[ci][si] !== null) {
                        continue; // Already assigned
                    }

                    const [d, s] = course.time_slots[si];
                    let placed = false;

                    // Try rooms by priority order
                    for (const room of allRooms) {
                        if (roomAvailable(d, s, room.name)) {
                            assignment[ci][si] = room.name;
                            setRoomOccupancy(d, s, room.name, 1);
                            console.log(`  Assigned ${course.name} session ${si} to ${room.name} (priority ${room.priority})`);
                            placed = true;
                            break;
                        }
                    }

                    if (!placed) {
                        assignment[ci][si] = "UNASSIGNED";
                        console.warn(`  Could not assign ${course.name} session ${si} - marked UNASSIGNED`);
                    }
                }
            }

            // PASS C: Try to satisfy remaining at_least_once by swapping
            console.log('PASS C: Trying to satisfy remaining obligations by swapping...');

            const priorityIndex = (roomName) => {
                const room = allRooms.find(r => r.name === roomName);
                return room ? allRooms.indexOf(room) : 999;
            };

            for (let ci = 0; ci < this.courses.length; ci++) {
                const course = this.courses[ci];
                const sessions = course.time_slots.map((slot, si) => ({ si, d: slot[0], s: slot[1] }));

                for (const room of obligatoryRooms) {
                    const roomName = room.name;

                    // Check if already satisfied
                    if (assignment[ci].some(assignedRoom => assignedRoom === roomName)) {
                        continue;
                    }

                    // Try to swap from a lower priority room
                    for (const session of sessions) {
                        const { si, d, s } = session;
                        const currentRoom = assignment[ci][si];

                        if (currentRoom === roomName || currentRoom === 'UNASSIGNED') {
                            continue;
                        }

                        // Check if we can swap
                        if (roomAvailable(d, s, roomName)) {
                            const currentPriorityIdx = priorityIndex(currentRoom);
                            const targetPriorityIdx = priorityIndex(roomName);

                            // Only swap if target has higher priority OR current is optional
                            const currentIsOptional = optionalRooms.some(r => r.name === currentRoom);

                            if (targetPriorityIdx < currentPriorityIdx || currentIsOptional) {
                                // Make the swap
                                setRoomOccupancy(d, s, currentRoom, 0);
                                assignment[ci][si] = roomName;
                                setRoomOccupancy(d, s, roomName, 1);
                                console.log(`  Swapped ${course.name} session ${si} from ${currentRoom} to ${roomName}`);
                                break;
                            }
                        }
                    }
                }
            }

            // Build result courses
            console.log('Building result...');
            const resultCourses = this.courses.map((course, ci) => {
                const classroomSlots = course.time_slots.map((slot, si) => [
                    slot[0], slot[1], assignment[ci][si]
                ]);

                return {
                    name: course.name,
                    color_code: course.color_code,
                    time_slots: course.time_slots,
                    classroom_slots: classroomSlots
                };
            });

            // Calculate statistics
            const stats = this.calculateStats(resultCourses, obligatoryRooms);

            console.log('Solver completed. Violations:', stats.violations.length);

            return {
                success: true,
                courses: resultCourses,
                stats: stats
            };
        },

        calculateStats(courses, mandatoryRooms) {
            const classroomUsage = {};
            const dailyUsage = {};
            const perCourseUsage = {};
            const violations = [];

            // Initialize
            this.classrooms.forEach(room => {
                classroomUsage[room.name] = 0;
            });

            courses.forEach(course => {
                const courseUsage = {};
                const usedRooms = new Set();

                course.classroom_slots.forEach(slot => {
                    const [weekday, slot_index, room_name] = slot;

                    if (room_name && room_name !== "UNASSIGNED") {
                        classroomUsage[room_name] = (classroomUsage[room_name] || 0) + 1;
                        courseUsage[room_name] = (courseUsage[room_name] || 0) + 1;
                        usedRooms.add(room_name);

                        if (!dailyUsage[weekday]) dailyUsage[weekday] = {};
                        dailyUsage[weekday][room_name] = (dailyUsage[weekday][room_name] || 0) + 1;
                    }
                });

                perCourseUsage[course.name] = courseUsage;

                // Check mandatory violations
                mandatoryRooms.forEach(room => {
                    if (!usedRooms.has(room.name)) {
                        violations.push({
                            course: course.name,
                            missing_room: room.name
                        });
                    }
                });
            });

            return {
                classroom_usage: classroomUsage,
                daily_usage: Object.fromEntries(
                    Object.entries(dailyUsage).map(([k, v]) => [k.toString(), v])
                ),
                per_course_usage: perCourseUsage,
                violations: violations
            };
        },

        // PDF generation
        async downloadPDF() {
            if (!this.assignedSchedule) return;

            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();

            doc.setFontSize(20);
            doc.text('Horario Escolar - Asignación Exacta', 20, 30);

            doc.setFontSize(10);
            doc.text(`Generado: ${new Date().toLocaleString('es-ES')}`, 20, 40);

            let yPos = 60;

            // Feasibility status
            doc.setFontSize(12);
            doc.text('Estado de Factibilidad:', 20, yPos);
            yPos += 7;
            doc.setFontSize(10);
            doc.text(`✓ ${this.feasibilityMessage}`, 25, yPos);
            yPos += 15;

            // Legend
            doc.setFontSize(14);
            doc.text('Leyenda de Aulas (Por Prioridad)', 20, yPos);
            yPos += 10;

            doc.setFontSize(10);
            this.classrooms.forEach((room, index) => {
                doc.text(`${index + 1}. ${room.name} (${room.obligatory_type})`, 25, yPos);
                yPos += 7;
            });

            yPos += 10;

            // Stats
            doc.setFontSize(14);
            doc.text('Estadísticas', 20, yPos);
            yPos += 10;

            doc.setFontSize(10);
            const stats = this.assignedSchedule.stats;

            doc.text('Uso por Aula:', 25, yPos);
            yPos += 7;
            Object.entries(stats.classroom_usage).forEach(([room, count]) => {
                doc.text(`  • ${room}: ${count} sesiones`, 30, yPos);
                yPos += 6;
            });

            yPos += 10;

            if (stats.violations.length > 0) {
                doc.setTextColor(255, 0, 0);
                doc.text('⚠ Restricciones no cumplidas:', 25, yPos);
                yPos += 7;
                stats.violations.forEach(violation => {
                    doc.text(`  • ${violation.course}: falta ${violation.missing_room}`, 30, yPos);
                    yPos += 6;
                });
                doc.setTextColor(0, 0, 0);
            } else {
                doc.setTextColor(0, 150, 0);
                doc.text('✓ Todas las restricciones cumplidas', 25, yPos);
                doc.setTextColor(0, 0, 0);
                yPos += 7;
            }

            // Detailed schedule
            doc.addPage();
            yPos = 30;

            doc.setFontSize(16);
            doc.text('Horario Detallado por Curso', 20, yPos);
            yPos += 20;

            doc.setFontSize(10);
            this.assignedSchedule.courses.forEach(course => {
                doc.setFontSize(12);
                doc.text(`${course.name}:`, 20, yPos);
                yPos += 7;

                doc.setFontSize(10);
                course.classroom_slots.forEach(slot => {
                    const [weekday, slot_index, room_name] = slot;
                    const dayName = this.weekdays[weekday] || `Día ${weekday}`;
                    const timeSlot = this.timeSlots.find(ts => 
                        ts.weekday === weekday && ts.slot_index === slot_index
                    );
                    const time = timeSlot ? `${timeSlot.start_time}-${timeSlot.end_time}` : `Tramo ${slot_index}`;

                    doc.text(`  • ${dayName} ${time} → ${room_name}`, 25, yPos);
                    yPos += 6;
                });

                yPos += 8;

                if (yPos > 270) {
                    doc.addPage();
                    yPos = 30;
                }
            });

            doc.save('horario-exacto.pdf');
        },

        // Utility methods
        getRandomColor() {
            const colors = ['#2563eb', '#059669', '#dc2626', '#d97706', '#7c3aed', '#0891b2'];
            return colors[Math.floor(Math.random() * colors.length)];
        },

        getContrastColor(hexColor) {
            const r = parseInt(hexColor.slice(1, 3), 16);
            const g = parseInt(hexColor.slice(3, 5), 16);
            const b = parseInt(hexColor.slice(5, 7), 16);
            const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
            return luminance > 0.6 ? '#000000' : '#FFFFFF';
        },

        shuffleArray(array) {
            for (let i = array.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [array[i], array[j]] = [array[j], array[i]];
            }
            return array;
        },

        showStatus(message, type) {
            this.statusMessage = message;
            this.statusType = type;
            setTimeout(() => {
                this.statusMessage = '';
            }, 5000);
        },

        showCellDialog(weekday, slotIndex) {
            console.log(`Clicked cell: Day ${weekday}, Slot ${slotIndex}`);
        },

        // Initialize drag & drop for classrooms
        initDragDrop() {
            this.$nextTick(() => {
                const el = document.getElementById('classrooms-sortable');
                if (el && window.Sortable) {
                    Sortable.create(el, {
                        animation: 150,
                        ghostClass: 'dragging',
                        onEnd: (evt) => {
                            const oldIndex = evt.oldIndex;
                            const newIndex = evt.newIndex;

                            // Reorder array
                            const item = this.classrooms.splice(oldIndex, 1)[0];
                            this.classrooms.splice(newIndex, 0, item);

                            this.updatePriorities();
                        }
                    });
                }
            });
        }
    },

    mounted() {
        // Initialize drag & drop
        this.initDragDrop();

        // Initialize with sample data
        this.addTimeSlot();
        this.timeSlots[0].weekday = 0;
        this.timeSlots[0].start_time = '09:00';
        this.timeSlots[0].end_time = '10:00';

        this.addTimeSlot();
        this.timeSlots[1].weekday = 0;
        this.timeSlots[1].start_time = '10:00';
        this.timeSlots[1].end_time = '11:00';

        this.addTimeSlot();
        this.timeSlots[2].weekday = 1;
        this.timeSlots[2].start_time = '09:00';
        this.timeSlots[2].end_time = '10:00';

        this.checkFeasibility();
    }
}).mount('#app');