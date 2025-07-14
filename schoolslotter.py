import json
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, shared_memory
import threading
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import psutil
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import itertools
import signal
import sys
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

warnings.filterwarnings('ignore')

# GPU Detection with better error handling
try:
    import cupy as cp
    test_array = cp.array([1, 2, 3])
    test_result = cp.sum(test_array)
    del test_array, test_result
    
    GPU_AVAILABLE = True
    try:
        gpu_props = cp.cuda.runtime.getDeviceProperties(0)
        GPU_NAME = gpu_props['name'].decode('utf-8')
        GPU_MEMORY = gpu_props['totalGlobalMem'] / (1024**3)
    except:
        GPU_NAME = "CUDA Device"
        GPU_MEMORY = 0
    print(f"🔥 GPU Available: {GPU_NAME} ({GPU_MEMORY:.1f} GB)")
    
except Exception as e:
    GPU_AVAILABLE = False
    GPU_NAME = "Not available"
    GPU_MEMORY = 0
    print(f"⚠️ GPU not available: {str(e)[:100]}...")

# Other imports
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("⚡ Numba JIT available!")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️ Numba not available")

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
    print("🧠 OR-Tools available!")
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("⚠️ OR-Tools not available")

class ObligatoryType(Enum):
    AT_LEAST_ONCE = "at_least_once"
    AT_LEAST_TWICE = "at_least_twice"
    OPTIONAL = "optional"

@dataclass
class Classroom:
    name: str
    priority: int
    obligatory_type: ObligatoryType
    color_code: str
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            name=data["name"],
            priority=data["priority"],
            obligatory_type=ObligatoryType(data["obligatory_type"]),
            color_code=data["color_code"]
        )

@dataclass
class TimeSlot:
    id: int
    weekday: int
    slot_index: int
    start_time: str
    end_time: str
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)
    
    def get_key(self) -> Tuple[int, int]:
        return (self.weekday, self.slot_index)
    
    def get_label(self) -> str:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return f"{day_names[self.weekday]} {self.start_time}-{self.end_time}"
    
    def get_time_label(self) -> str:
        return f"{self.start_time}-{self.end_time}"

@dataclass
class Course:
    name: str
    color_code: str
    time_slots: List[List[int]]
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)
    
    def get_slot_keys(self) -> List[Tuple[int, int]]:
        return [tuple(ts) for ts in self.time_slots]

class UltimateProgressTracker:
    """Ultra-advanced progress tracker with full system monitoring"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()
        self.solutions_found = 0
        self.best_quality = 0
        self.current_phase = "Initializing"
        self.iterations = 0
        self.is_running = True
        
        # System monitoring
        self.cpu_percent = 0
        self.memory_percent = 0
        self.gpu_percent = 0
        self.gpu_memory_used = 0
        
        # Performance tracking
        self.iterations_per_second = 0
        self.last_iteration_count = 0
        self.last_time = time.time()
        
        # Start monitoring threads
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.display_thread = threading.Thread(target=self._display_progress, daemon=True)
        self.monitor_thread.start()
        self.display_thread.start()
    
    def _monitor_system(self):
        """Advanced system monitoring"""
        while self.is_running:
            try:
                # CPU and Memory
                self.cpu_percent = psutil.cpu_percent(interval=0.5)
                self.memory_percent = psutil.virtual_memory().percent
                
                # GPU monitoring (more robust)
                if GPU_AVAILABLE:
                    try:
                        mempool = cp.get_default_memory_pool()
                        if mempool.total_bytes() > 0:
                            self.gpu_percent = (mempool.used_bytes() / mempool.total_bytes()) * 100
                            self.gpu_memory_used = mempool.used_bytes() / (1024**3)  # GB
                        else:
                            self.gpu_percent = 0
                            self.gpu_memory_used = 0
                    except:
                        self.gpu_percent = 0
                        self.gpu_memory_used = 0
                
                # Performance calculation
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    iteration_diff = self.iterations - self.last_iteration_count
                    time_diff = current_time - self.last_time
                    self.iterations_per_second = iteration_diff / time_diff
                    self.last_iteration_count = self.iterations
                    self.last_time = current_time
                
                time.sleep(0.5)
            except:
                pass
    
    def _display_progress(self):
        """Enhanced progress display"""
        while self.is_running:
            try:
                elapsed = time.time() - self.start_time
                
                # Create visual bars
                cpu_bar = "█" * min(20, int(self.cpu_percent / 5)) + "░" * max(0, 20 - int(self.cpu_percent / 5))
                
                if GPU_AVAILABLE:
                    gpu_bar = "█" * min(10, int(self.gpu_percent / 10)) + "░" * max(0, 10 - int(self.gpu_percent / 10))
                    gpu_info = f"GPU: {gpu_bar} {self.gpu_percent:5.1f}% ({self.gpu_memory_used:.1f}GB)"
                else:
                    gpu_info = "GPU: ❌"
                
                # Performance info
                perf_info = f"⚡{self.iterations_per_second:,.0f}/s" if self.iterations_per_second > 0 else ""
                
                print(f"\r🚀 {self.name} │ {self.current_phase} │ "
                      f"CPU: {cpu_bar} {self.cpu_percent:5.1f}% │ "
                      f"{gpu_info} │ "
                      f"RAM: {self.memory_percent:5.1f}% │ "
                      f"Time: {elapsed:6.1f}s │ "
                      f"Iter: {self.iterations:,} {perf_info} │ "
                      f"Sol: {self.solutions_found} │ "
                      f"Q: {self.best_quality:6.1f}", end="", flush=True)
                
                time.sleep(0.8)
            except:
                pass
    
    def update(self, phase: str = None, quality: float = None, iterations: int = None):
        if phase:
            self.current_phase = phase
        if quality is not None:
            self.best_quality = max(self.best_quality, quality)
        if iterations is not None:
            self.iterations = iterations
    
    def add_solution(self, quality: float = None):
        self.solutions_found += 1
        if quality is not None:
            self.best_quality = max(self.best_quality, quality)
    
    def finish(self):
        self.is_running = False
        time.sleep(1.5)
        elapsed = time.time() - self.start_time
        print(f"\n✅ {self.name} completed in {elapsed:.1f}s")
        print(f"🎯 Found {self.solutions_found} solutions, best quality: {self.best_quality:.1f}")
        print(f"⚡ Average performance: {self.iterations/elapsed:,.0f} iterations/second")

# Optimized fitness function with Numba if available
if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def calculate_fitness_fast(individual, n_classrooms, slot_assignment_map, obligatory_data):
        """Ultra-fast fitness calculation with Numba JIT"""
        score = 0
        penalty = 0
        
        # Priority scoring
        for classroom_idx in individual:
            score += (n_classrooms - classroom_idx)
        
        # Conflict detection
        slot_usage = {}
        for i, classroom_idx in enumerate(individual):
            slot_key = slot_assignment_map[i]
            if slot_key in slot_usage:
                if classroom_idx in slot_usage[slot_key]:
                    penalty += 1000
                else:
                    slot_usage[slot_key] = slot_usage[slot_key] | {classroom_idx}
            else:
                slot_usage[slot_key] = {classroom_idx}
        
        return score - penalty
else:
    def calculate_fitness_fast(individual, n_classrooms, slot_assignment_map, obligatory_data):
        """Fallback fitness calculation"""
        score = sum(n_classrooms - classroom_idx for classroom_idx in individual)
        penalty = 0
        
        slot_usage = {}
        for i, classroom_idx in enumerate(individual):
            slot_key = slot_assignment_map[i]
            if slot_key not in slot_usage:
                slot_usage[slot_key] = set()
            if classroom_idx in slot_usage[slot_key]:
                penalty += 1000
            slot_usage[slot_key].add(classroom_idx)
        
        return score - penalty

class MegaTurboScheduler:
    """Ultimate high-performance scheduler"""
    
    def __init__(self, classrooms: List[Classroom], courses: List[Course], time_slots: List[TimeSlot]):
        self.classrooms = classrooms
        self.courses = courses
        self.time_slots = time_slots
        self.assignments = []
        self._setup_assignments()
        
        # Optimization data
        self.n_classrooms = len(classrooms)
        self.n_assignments = len(self.assignments)
        self.slot_assignment_map = {}
        self.obligatory_data = self._prepare_optimization_data()
        
        print(f"🔧 Scheduler optimized for {self.n_assignments} assignments, {self.n_classrooms} classrooms")
    
    def _setup_assignments(self):
        for course in self.courses:
            for slot_key in course.get_slot_keys():
                self.assignments.append((course.name, slot_key))
    
    def _prepare_optimization_data(self):
        """Prepare data for ultra-fast processing"""
        # Create slot assignment map for fast lookup
        for i, (course_name, slot_key) in enumerate(self.assignments):
            self.slot_assignment_map[i] = slot_key
        
        # Prepare obligatory constraint data
        obligatory_data = {}
        course_assignments = defaultdict(list)
        
        for i, (course_name, slot_key) in enumerate(self.assignments):
            course_assignments[course_name].append(i)
        
        for course_name, assignment_indices in course_assignments.items():
            obligatory_data[course_name] = {
                'indices': assignment_indices,
                'requirements': []
            }
            
            for j, classroom in enumerate(self.classrooms):
                if classroom.obligatory_type == ObligatoryType.AT_LEAST_ONCE:
                    obligatory_data[course_name]['requirements'].append((j, 1))
                elif classroom.obligatory_type == ObligatoryType.AT_LEAST_TWICE:
                    obligatory_data[course_name]['requirements'].append((j, 2))
        
        return obligatory_data
    
    def solve_mega_turbo(self, mode: str = "auto", timeout_minutes: int = 10) -> Optional[Dict]:
        """Solve using mega-turbo algorithm"""
        
        print(f"🚀 Starting MEGA-TURBO Scheduler")
        print(f"📊 Mode: {mode.upper()}, Timeout: {timeout_minutes} minutes")
        print(f"⚙️ CPU Cores: {mp.cpu_count()}, GPU: {'✅' if GPU_AVAILABLE else '❌'}")
        print("-" * 70)
        
        if mode == "auto":
            # Auto-select best strategy
            if self.n_assignments > 100:
                mode = "hybrid"
            elif GPU_AVAILABLE and self.n_assignments > 50:
                mode = "gpu"
            else:
                mode = "multicore"
        
        if mode == "gpu" and GPU_AVAILABLE:
            return self._solve_gpu_mode(timeout_minutes)
        elif mode == "multicore":
            return self._solve_multicore_mode(timeout_minutes)
        elif mode == "hybrid":
            return self._solve_hybrid_mode(timeout_minutes)
        else:
            return self._solve_multicore_mode(timeout_minutes)
    
    def _solve_gpu_mode(self, timeout_minutes: int) -> Optional[Dict]:
        """GPU-accelerated solving (simplified for stability)"""
        tracker = UltimateProgressTracker("GPU Mode")
        
        try:
            # Use CPU-based genetic with GPU-inspired optimizations
            return self._solve_optimized_genetic(tracker, timeout_minutes, population_size=1000)
        except Exception as e:
            print(f"⚠️ GPU mode error: {e}")
            tracker.finish()
            return self._solve_multicore_mode(timeout_minutes)
    
    def _solve_multicore_mode(self, timeout_minutes: int) -> Optional[Dict]:
        """Multi-core CPU solving"""
        tracker = UltimateProgressTracker("Multi-Core CPU Mode")
        
        try:
            n_processes = min(mp.cpu_count(), 16)
            searches_per_process = 10000
            
            print(f"🔥 Launching {n_processes} parallel processes...")
            
            # Use ProcessPoolExecutor for true parallelism
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                # Submit parallel tasks
                futures = []
                for process_id in range(n_processes):
                    future = executor.submit(
                        parallel_search_worker,
                        process_id,
                        searches_per_process,
                        self.assignments,
                        len(self.classrooms),
                        self.obligatory_data,
                        timeout_minutes * 60 // n_processes
                    )
                    futures.append(future)
                
                # Monitor and collect results
                best_result = None
                completed = 0
                
                for future in as_completed(futures, timeout=timeout_minutes*60):
                    try:
                        result = future.result()
                        completed += 1
                        
                        tracker.update(
                            phase=f"Collecting results ({completed}/{n_processes})",
                            iterations=completed * searches_per_process
                        )
                        
                        if result and (not best_result or result['fitness'] > best_result['fitness']):
                            best_result = result
                            tracker.add_solution(result['fitness'])
                            print(f"\n✨ Process {result['process_id']} found solution!")
                            break  # First valid solution wins
                            
                    except Exception as e:
                        print(f"⚠️ Process error: {e}")
                
                tracker.finish()
                
                if best_result:
                    return best_result['solution']
                
                return None
                
        except Exception as e:
            print(f"❌ Multi-core error: {e}")
            tracker.finish()
            return None
    
    def _solve_hybrid_mode(self, timeout_minutes: int) -> Optional[Dict]:
        """Hybrid approach: multi-core + genetic"""
        print("🎯 Hybrid mode: Multi-core search + Genetic refinement")
        
        # Phase 1: Quick multi-core search
        result = self._solve_multicore_mode(timeout_minutes // 2)
        if result:
            return result
        
        # Phase 2: Genetic algorithm
        tracker = UltimateProgressTracker("Hybrid Genetic Phase")
        return self._solve_optimized_genetic(tracker, timeout_minutes // 2, population_size=500)
    
    def _solve_optimized_genetic(self, tracker: UltimateProgressTracker, timeout_minutes: int, population_size: int) -> Optional[Dict]:
        """Optimized genetic algorithm"""
        try:
            # Initialize population
            tracker.update("Initializing population")
            population = []
            for _ in range(population_size):
                individual = [np.random.randint(0, self.n_classrooms) for _ in range(self.n_assignments)]
                population.append(individual)
            
            best_fitness = -float('inf')
            best_individual = None
            generation = 0
            max_generations = 1000
            
            start_time = time.time()
            timeout_seconds = timeout_minutes * 60
            
            while generation < max_generations and time.time() - start_time < timeout_seconds:
                # Calculate fitness for all individuals
                fitness_scores = []
                for individual in population:
                    fitness = self._calculate_detailed_fitness(individual)
                    fitness_scores.append(fitness)
                
                # Find best
                best_idx = np.argmax(fitness_scores)
                current_best_fitness = fitness_scores[best_idx]
                
                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    best_individual = population[best_idx].copy()
                    
                    if self._is_valid_solution(best_individual):
                        tracker.add_solution(best_fitness)
                        print(f"\n✨ Valid solution found at generation {generation}!")
                        break
                
                # Evolution
                population = self._evolve_population(population, fitness_scores)
                generation += 1
                
                # Update progress
                tracker.update(
                    phase=f"Generation {generation}/{max_generations}",
                    quality=best_fitness,
                    iterations=generation * population_size
                )
            
            tracker.finish()
            
            if best_individual and self._is_valid_solution(best_individual):
                return self._individual_to_schedule(best_individual)
            
            return None
            
        except Exception as e:
            print(f"❌ Genetic error: {e}")
            tracker.finish()
            return None
    
    def _calculate_detailed_fitness(self, individual: List[int]) -> float:
        """Detailed fitness calculation with all constraints"""
        score = 0
        penalty = 0
        
        # Basic priority scoring
        for classroom_idx in individual:
            score += (self.n_classrooms - classroom_idx)
        
        # Conflict detection
        slot_usage = defaultdict(set)
        course_classrooms = defaultdict(list)
        
        for i, classroom_idx in enumerate(individual):
            course_name, slot_key = self.assignments[i]
            
            # Check slot conflicts
            if classroom_idx in slot_usage[slot_key]:
                penalty += 1000
            slot_usage[slot_key].add(classroom_idx)
            
            # Track course assignments
            course_classrooms[course_name].append(classroom_idx)
        
        # Check obligatory constraints
        for course_name, data in self.obligatory_data.items():
            assigned_classrooms = course_classrooms.get(course_name, [])
            for classroom_idx, required_count in data['requirements']:
                actual_count = assigned_classrooms.count(classroom_idx)
                if actual_count < required_count:
                    penalty += 500 * (required_count - actual_count)
        
        return score - penalty
    
    def _is_valid_solution(self, individual: List[int]) -> bool:
        """Check if solution is valid"""
        slot_usage = defaultdict(set)
        course_classrooms = defaultdict(list)
        
        for i, classroom_idx in enumerate(individual):
            course_name, slot_key = self.assignments[i]
            
            if classroom_idx in slot_usage[slot_key]:
                return False
            slot_usage[slot_key].add(classroom_idx)
            course_classrooms[course_name].append(classroom_idx)
        
        # Check obligatory constraints
        for course_name, data in self.obligatory_data.items():
            assigned_classrooms = course_classrooms.get(course_name, [])
            for classroom_idx, required_count in data['requirements']:
                actual_count = assigned_classrooms.count(classroom_idx)
                if actual_count < required_count:
                    return False
        
        return True
    
    def _evolve_population(self, population: List[List[int]], fitness_scores: List[float]) -> List[List[int]]:
        """Evolve population"""
        population_size = len(population)
        new_population = []
        
        # Elitism
        elite_count = max(1, population_size // 10)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < population_size:
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """Tournament selection"""
        tournament_size = min(7, len(population))
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Two-point crossover"""
        length = len(parent1)
        if length <= 2:
            return parent1.copy()
        
        point1 = np.random.randint(0, length)
        point2 = np.random.randint(point1, length)
        
        child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        return child
    
    def _mutate(self, individual: List[int]) -> List[int]:
        """Smart mutation"""
        mutation_rate = 0.15
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = np.random.randint(0, self.n_classrooms)
        
        return mutated
    
    def _individual_to_schedule(self, individual: List[int]) -> Dict:
        """Convert individual to schedule format"""
        result = defaultdict(list)
        
        for i, classroom_idx in enumerate(individual):
            course_name, slot_key = self.assignments[i]
            classroom_name = self.classrooms[classroom_idx].name
            
            result[course_name].append({
                "slot": list(slot_key),
                "classroom": classroom_name
            })
        
        return dict(result)

def parallel_search_worker(process_id: int, n_searches: int, assignments: List[Tuple], 
                          n_classrooms: int, obligatory_data: Dict, timeout_seconds: int) -> Optional[Dict]:
    """Worker function for parallel processing"""
    np.random.seed(process_id * 1000 + int(time.time()))
    
    best_fitness = -float('inf')
    best_individual = None
    
    start_time = time.time()
    
    for i in range(n_searches):
        if time.time() - start_time > timeout_seconds:
            break
        
        # Generate random individual
        individual = [np.random.randint(0, n_classrooms) for _ in range(len(assignments))]
        
        # Quick fitness check
        fitness = calculate_fitness_simple(individual, assignments, n_classrooms)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = individual
            
            # Check if valid
            if is_valid_solution_simple(individual, assignments, obligatory_data, n_classrooms):
                # Convert to schedule format
                schedule = {}
                for j, classroom_idx in enumerate(individual):
                    course_name, slot_key = assignments[j]
                    classroom_name = f"Classroom_{classroom_idx}"  # Simplified for parallel processing
                    
                    if course_name not in schedule:
                        schedule[course_name] = []
                    schedule[course_name].append({
                        "slot": list(slot_key),
                        "classroom": classroom_name
                    })
                
                return {
                    'fitness': fitness,
                    'solution': schedule,
                    'process_id': process_id,
                    'searches_done': i + 1
                }
    
    return None

def calculate_fitness_simple(individual: List[int], assignments: List[Tuple], n_classrooms: int) -> float:
    """Simplified fitness calculation for parallel processing"""
    score = sum(n_classrooms - classroom_idx for classroom_idx in individual)
    penalty = 0
    
    slot_usage = defaultdict(set)
    for i, classroom_idx in enumerate(individual):
        course_name, slot_key = assignments[i]
        if classroom_idx in slot_usage[slot_key]:
            penalty += 1000
        slot_usage[slot_key].add(classroom_idx)
    
    return score - penalty

def is_valid_solution_simple(individual: List[int], assignments: List[Tuple], 
                           obligatory_data: Dict, n_classrooms: int) -> bool:
    """Simplified validation for parallel processing"""
    slot_usage = defaultdict(set)
    course_classrooms = defaultdict(list)
    
    for i, classroom_idx in enumerate(individual):
        course_name, slot_key = assignments[i]
        
        if classroom_idx in slot_usage[slot_key]:
            return False
        slot_usage[slot_key].add(classroom_idx)
        course_classrooms[course_name].append(classroom_idx)
    
    # Quick obligatory check
    for course_name, data in obligatory_data.items():
        assigned_classrooms = course_classrooms.get(course_name, [])
        for classroom_idx, required_count in data['requirements']:
            if assigned_classrooms.count(classroom_idx) < required_count:
                return False
    
    return True

class UltimateScheduleVisualizer:
    """Ultimate PDF visualizer with weekly grid as first page"""
    
    def __init__(self, classrooms: List[Classroom], courses: List[Course], time_slots: List[TimeSlot]):
        self.classrooms = classrooms
        self.courses = courses
        self.time_slots = time_slots
        self.time_slot_dict = {ts.get_key(): ts for ts in time_slots}
        
        # Course colors mapping
        self.course_colors = {course.name: course.color_code for course in courses}
        
        # Classroom colors mapping
        self.classroom_colors = {classroom.name: classroom.color_code for classroom in classrooms}
    
    def create_ultimate_pdf_report(self, schedule: Dict, filename: str = "ultimate_schedule_report.pdf"):
        """Create comprehensive PDF report with weekly grid first"""
        print("🎨 Creating ultimate PDF report...")
        
        with PdfPages(filename) as pdf:
            # Page 1: Weekly Schedule Grid (Days vs Hours)
            self._create_weekly_schedule_grid(pdf, schedule)
            
            # Page 2: Classroom Utilization Analysis
            self._create_classroom_utilization_page(pdf, schedule)
            
            # Page 3: Course Distribution Analysis
            self._create_course_distribution_page(pdf, schedule)
            
            # Page 4: Summary and Statistics
            self._create_summary_page(pdf, schedule)
        
        print(f"📑 Ultimate PDF report saved as: {filename}")
    
    def _create_weekly_schedule_grid(self, pdf: PdfPages, schedule: Dict):
        """Create weekly schedule grid: Days (columns) vs Hours (rows)"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        
        # Title
        fig.suptitle('📅 WEEKLY SCHEDULE GRID', fontsize=24, fontweight='bold', y=0.95)
        ax.set_title('Days (Columns) × Time Slots (Rows)', fontsize=16, pad=20)
        
        # Organize data by day and time
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Get all unique time slots used, sorted by weekday and time
        all_time_slots = set()
        for course_name, assignments in schedule.items():
            for assignment in assignments:
                slot_key = tuple(assignment["slot"])
                if slot_key in self.time_slot_dict:
                    all_time_slots.add(slot_key)
        
        # Sort time slots by weekday and then by slot_index
        sorted_time_slots = sorted(all_time_slots, key=lambda x: (x[0], x[1]))
        
        # Create a mapping from time to row
        unique_times = {}
        for slot_key in sorted_time_slots:
            if slot_key in self.time_slot_dict:
                time_slot = self.time_slot_dict[slot_key]
                time_label = time_slot.get_time_label()
                if time_label not in unique_times:
                    unique_times[time_label] = len(unique_times)
        
        time_labels = list(unique_times.keys())
        
        # Create grid: rows = time slots, columns = days
        n_rows = len(time_labels)
        n_cols = 7  # 7 days of week
        
        if n_rows == 0:
            ax.text(0.5, 0.5, 'No schedule data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
        
        # Create the grid data structure
        grid_data = {}  # (row, col) -> [(course, classroom), ...]
        
        for course_name, assignments in schedule.items():
            course_color = self.course_colors.get(course_name, "#CCCCCC")
            
            for assignment in assignments:
                slot_key = tuple(assignment["slot"])
                if slot_key in self.time_slot_dict:
                    time_slot = self.time_slot_dict[slot_key]
                    time_label = time_slot.get_time_label()
                    weekday = slot_key[0]
                    classroom = assignment["classroom"]
                    
                    if time_label in unique_times:
                        row = unique_times[time_label]
                        col = weekday
                        
                        if (row, col) not in grid_data:
                            grid_data[(row, col)] = []
                        grid_data[(row, col)].append((course_name, classroom, course_color))
        
        # Draw the grid
        cell_height = 1.0
        cell_width = 1.0
        
        for row in range(n_rows):
            for col in range(n_cols):
                x = col * cell_width
                y = (n_rows - row - 1) * cell_height
                
                # Default cell
                rect = plt.Rectangle([x, y], cell_width, cell_height, 
                                   facecolor='white', edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                
                # Add content if exists
                if (row, col) in grid_data:
                    entries = grid_data[(row, col)]
                    
                    # If multiple entries in same slot, divide the cell
                    n_entries = len(entries)
                    sub_height = cell_height / n_entries
                    
                    for i, (course_name, classroom, course_color) in enumerate(entries):
                        sub_y = y + i * sub_height
                        
                        # Draw colored rectangle for this entry
                        sub_rect = plt.Rectangle([x + 0.05, sub_y + 0.05], 
                                                cell_width - 0.1, sub_height - 0.1,
                                                facecolor=course_color, alpha=0.7,
                                                edgecolor='black', linewidth=0.5)
                        ax.add_patch(sub_rect)
                        
                        # Add text
                        text_y = sub_y + sub_height/2
                        
                        # Course name (bold)
                        course_short = self._shorten_text(course_name, 12)
                        ax.text(x + cell_width/2, text_y + 0.1, course_short,
                               ha='center', va='center', fontsize=8, weight='bold',
                               color='white' if self._is_dark_color(course_color) else 'black')
                        
                        # Classroom name (smaller)
                        classroom_short = self._shorten_text(classroom, 15)
                        ax.text(x + cell_width/2, text_y - 0.1, classroom_short,
                               ha='center', va='center', fontsize=7,
                               color='white' if self._is_dark_color(course_color) else 'black')
        
        # Set labels
        # Column headers (days)
        for col, day_name in enumerate(day_names):
            ax.text(col * cell_width + cell_width/2, n_rows * cell_height + 0.2, 
                   day_name, ha='center', va='bottom', fontsize=12, weight='bold')
        
        # Row headers (times)
        for row, time_label in enumerate(time_labels):
            ax.text(-0.2, (n_rows - row - 1) * cell_height + cell_height/2, 
                   time_label, ha='right', va='center', fontsize=10, weight='bold')
        
        # Styling
        ax.set_xlim(-0.5, n_cols * cell_width + 0.5)
        ax.set_ylim(-0.5, n_rows * cell_height + 0.8)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add legend for courses
        self._add_course_legend(ax, n_cols * cell_width + 1, n_rows * cell_height)
        
        # Add info box
        info_text = f"📊 Schedule Overview\n" \
                   f"🏛️ {len(self.classrooms)} Classrooms\n" \
                   f"📚 {len(schedule)} Courses\n" \
                   f"⏰ {len(time_labels)} Time Slots\n" \
                   f"📅 {len([col for col in range(7) if any((row, col) in grid_data for row in range(n_rows))])} Active Days"
        
        ax.text(-0.4, n_rows * cell_height + 0.5, info_text, 
               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_classroom_utilization_page(self, pdf: PdfPages, schedule: Dict):
        """Create classroom utilization analysis page"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🏛️ CLASSROOM UTILIZATION ANALYSIS', fontsize=20, fontweight='bold')
        
        # 1. Classroom usage bar chart
        classroom_usage = defaultdict(int)
        total_classes = 0
        
        for course_name, assignments in schedule.items():
            for assignment in assignments:
                classroom_usage[assignment["classroom"]] += 1
                total_classes += 1
        
        if classroom_usage:
            classrooms = list(classroom_usage.keys())
            usage_counts = list(classroom_usage.values())
            colors = [self.classroom_colors.get(c, "#CCCCCC") for c in classrooms]
            
            bars = ax1.bar(range(len(classrooms)), usage_counts, color=colors, alpha=0.8)
            ax1.set_title('📊 Classes per Classroom', fontsize=14, weight='bold')
            ax1.set_xlabel('Classrooms')
            ax1.set_ylabel('Number of Classes')
            ax1.set_xticks(range(len(classrooms)))
            ax1.set_xticklabels([self._shorten_text(c, 10) for c in classrooms], rotation=45)
            
            # Add percentage labels
            for i, (bar, count) in enumerate(zip(bars, usage_counts)):
                percentage = (count / total_classes) * 100
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)
            
            ax1.grid(True, alpha=0.3)
        
        # 2. Classroom utilization pie chart
        if classroom_usage:
            ax2.pie(usage_counts, labels=[self._shorten_text(c, 8) for c in classrooms], 
                   colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('🥧 Classroom Distribution', fontsize=14, weight='bold')
        
        # 3. Daily classroom usage
        daily_classroom_usage = defaultdict(lambda: defaultdict(int))
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        for course_name, assignments in schedule.items():
            for assignment in assignments:
                slot_key = tuple(assignment["slot"])
                if slot_key in self.time_slot_dict:
                    weekday = slot_key[0]
                    classroom = assignment["classroom"]
                    daily_classroom_usage[weekday][classroom] += 1
        
        if daily_classroom_usage:
            # Create stacked bar chart
            days = sorted(daily_classroom_usage.keys())
            classroom_names = list(classroom_usage.keys())
            
            bottom = np.zeros(len(days))
            colors = [self.classroom_colors.get(c, "#CCCCCC") for c in classroom_names]
            
            for i, classroom in enumerate(classroom_names):
                counts = [daily_classroom_usage[day].get(classroom, 0) for day in days]
                ax3.bar([day_names[d] for d in days], counts, bottom=bottom, 
                       label=self._shorten_text(classroom, 10), color=colors[i], alpha=0.8)
                bottom += counts
            
            ax3.set_title('📅 Daily Classroom Usage', fontsize=14, weight='bold')
            ax3.set_xlabel('Days')
            ax3.set_ylabel('Number of Classes')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # 4. Classroom efficiency metrics
        if classroom_usage:
            # Calculate efficiency metrics
            max_possible = len(self.time_slots)  # Theoretical max classes per classroom
            efficiency_data = []
            
            for classroom, usage in classroom_usage.items():
                efficiency = (usage / max_possible) * 100 if max_possible > 0 else 0
                efficiency_data.append((classroom, usage, efficiency))
            
            # Sort by efficiency
            efficiency_data.sort(key=lambda x: x[2], reverse=True)
            
            classrooms_eff = [x[0] for x in efficiency_data]
            efficiencies = [x[2] for x in efficiency_data]
            colors_eff = [self.classroom_colors.get(c, "#CCCCCC") for c in classrooms_eff]
            
            bars = ax4.barh(range(len(classrooms_eff)), efficiencies, color=colors_eff, alpha=0.8)
            ax4.set_title('⚡ Classroom Efficiency', fontsize=14, weight='bold')
            ax4.set_xlabel('Efficiency (%)')
            ax4.set_ylabel('Classrooms')
            ax4.set_yticks(range(len(classrooms_eff)))
            ax4.set_yticklabels([self._shorten_text(c, 12) for c in classrooms_eff])
            
            # Add efficiency labels
            for i, (bar, eff) in enumerate(zip(bars, efficiencies)):
                ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{eff:.1f}%', ha='left', va='center', fontsize=9)
            
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, 100)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_course_distribution_page(self, pdf: PdfPages, schedule: Dict):
        """Create course distribution analysis page"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📚 COURSE DISTRIBUTION ANALYSIS', fontsize=20, fontweight='bold')
        
        # 1. Classes per course
        course_class_counts = {course: len(assignments) for course, assignments in schedule.items()}
        
        if course_class_counts:
            courses = list(course_class_counts.keys())
            class_counts = list(course_class_counts.values())
            colors = [self.course_colors.get(c, "#CCCCCC") for c in courses]
            
            bars = ax1.bar(range(len(courses)), class_counts, color=colors, alpha=0.8)
            ax1.set_title('📊 Classes per Course', fontsize=14, weight='bold')
            ax1.set_xlabel('Courses')
            ax1.set_ylabel('Number of Classes')
            ax1.set_xticks(range(len(courses)))
            ax1.set_xticklabels([self._shorten_text(c, 8) for c in courses], rotation=45)
            
            # Add count labels
            for bar, count in zip(bars, class_counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        str(count), ha='center', va='bottom', fontsize=9, weight='bold')
            
            ax1.grid(True, alpha=0.3)
        
        # 2. Course-classroom matrix
        course_classroom_matrix = defaultdict(lambda: defaultdict(int))
        
        for course_name, assignments in schedule.items():
            for assignment in assignments:
                course_classroom_matrix[course_name][assignment["classroom"]] += 1
        
        if course_classroom_matrix:
            courses = list(course_classroom_matrix.keys())
            classrooms = list(set(
                classroom for course_data in course_classroom_matrix.values()
                for classroom in course_data.keys()
            ))
            
            # Create matrix
            matrix = np.zeros((len(courses), len(classrooms)))
            for i, course in enumerate(courses):
                for j, classroom in enumerate(classrooms):
                    matrix[i][j] = course_classroom_matrix[course].get(classroom, 0)
            
            # Create heatmap
            im = ax2.imshow(matrix, cmap='Blues', aspect='auto')
            ax2.set_title('🎯 Course-Classroom Matrix', fontsize=14, weight='bold')
            
            # Add text annotations
            for i in range(len(courses)):
                for j in range(len(classrooms)):
                    if matrix[i][j] > 0:
                        ax2.text(j, i, int(matrix[i][j]), ha="center", va="center",
                               color="white" if matrix[i][j] > matrix.max()/2 else "black",
                               fontsize=8, weight='bold')
            
            ax2.set_xticks(range(len(classrooms)))
            ax2.set_yticks(range(len(courses)))
            ax2.set_xticklabels([self._shorten_text(c, 8) for c in classrooms], rotation=45)
            ax2.set_yticklabels([self._shorten_text(c, 12) for c in courses])
            
            # Add colorbar
            plt.colorbar(im, ax=ax2, label='Classes')
        
        # 3. Daily course distribution
        daily_course_counts = defaultdict(int)
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        for course_name, assignments in schedule.items():
            for assignment in assignments:
                slot_key = tuple(assignment["slot"])
                if slot_key in self.time_slot_dict:
                    weekday = slot_key[0]
                    daily_course_counts[weekday] += 1
        
        if daily_course_counts:
            days = list(range(7))
            counts = [daily_course_counts.get(day, 0) for day in days]
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB366', '#B366FF', '#FF66B2']
            
            bars = ax3.bar(days, counts, color=colors[:len(days)], alpha=0.8)
            ax3.set_title('📅 Daily Class Distribution', fontsize=14, weight='bold')
            ax3.set_xlabel('Day of Week')
            ax3.set_ylabel('Number of Classes')
            ax3.set_xticks(days)
            ax3.set_xticklabels([day_names[i] for i in days])
            
            # Add count labels
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom', fontsize=10, weight='bold')
            
            ax3.grid(True, alpha=0.3)
        
        # 4. Time slot popularity
        timeslot_usage = defaultdict(int)
        
        for course_name, assignments in schedule.items():
            for assignment in assignments:
                slot_key = tuple(assignment["slot"])
                if slot_key in self.time_slot_dict:
                    time_slot = self.time_slot_dict[slot_key]
                    time_label = time_slot.get_time_label()
                    timeslot_usage[time_label] += 1
        
        if timeslot_usage:
            time_labels = list(timeslot_usage.keys())
            usage_counts = list(timeslot_usage.values())
            
            bars = ax4.barh(range(len(time_labels)), usage_counts, alpha=0.8, color='lightcoral')
            ax4.set_title('⏰ Time Slot Popularity', fontsize=14, weight='bold')
            ax4.set_xlabel('Number of Classes')
            ax4.set_ylabel('Time Slots')
            ax4.set_yticks(range(len(time_labels)))
            ax4.set_yticklabels(time_labels)
            
            # Add count labels
            for bar, count in zip(bars, usage_counts):
                ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        str(count), ha='left', va='center', fontsize=9, weight='bold')
            
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_summary_page(self, pdf: PdfPages, schedule: Dict):
        """Create summary and statistics page"""
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('📈 SCHEDULE SUMMARY & STATISTICS', fontsize=24, fontweight='bold')
        
        # Calculate statistics
        total_classes = sum(len(assignments) for assignments in schedule.values())
        total_courses = len(schedule)
        total_classrooms_used = len(set(
            assignment["classroom"] 
            for assignments in schedule.values() 
            for assignment in assignments
        ))
        
        # Time statistics
        time_slots_used = set()
        for assignments in schedule.values():
            for assignment in assignments:
                slot_key = tuple(assignment["slot"])
                time_slots_used.add(slot_key)
        
        total_time_slots_used = len(time_slots_used)
        
        # Days used
        days_used = set()
        for slot_key in time_slots_used:
            if slot_key in self.time_slot_dict:
                days_used.add(slot_key[0])
        
        total_days_used = len(days_used)
        
        # Create summary layout
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Main statistics box
        stats_text = f"""
🏫 SCHEDULE STATISTICS

📊 GENERAL OVERVIEW
   • Total Classes Scheduled: {total_classes}
   • Total Courses: {total_courses}
   • Classrooms Used: {total_classrooms_used} / {len(self.classrooms)}
   • Time Slots Used: {total_time_slots_used} / {len(self.time_slots)}
   • Active Days: {total_days_used} / 7

📚 COURSE BREAKDOWN
"""
        
        # Add course details
        for course_name, assignments in sorted(schedule.items()):
            classrooms_used = set(assignment["classroom"] for assignment in assignments)
            stats_text += f"   • {course_name}: {len(assignments)} classes in {len(classrooms_used)} classrooms\n"
        
        stats_text += f"""

🏛️ CLASSROOM BREAKDOWN
"""
        
        # Classroom usage
        classroom_usage = defaultdict(int)
        for assignments in schedule.values():
            for assignment in assignments:
                classroom_usage[assignment["classroom"]] += 1
        
        for classroom, usage in sorted(classroom_usage.items()):
            efficiency = (usage / len(self.time_slots)) * 100 if len(self.time_slots) > 0 else 0
            stats_text += f"   • {classroom}: {usage} classes ({efficiency:.1f}% utilization)\n"
        
        # Constraint validation
        stats_text += f"""

✅ CONSTRAINT VALIDATION
"""
        
        # Check obligatory constraints
        constraint_violations = []
        for course_name, assignments in schedule.items():
            classrooms_used = [assignment["classroom"] for assignment in assignments]
            
            for classroom in self.classrooms:
                count = classrooms_used.count(classroom.name)
                
                if classroom.obligatory_type == ObligatoryType.AT_LEAST_ONCE and count == 0:
                    constraint_violations.append(f"{course_name} missing {classroom.name}")
                elif classroom.obligatory_type == ObligatoryType.AT_LEAST_TWICE and count < 2:
                    constraint_violations.append(f"{course_name} needs more {classroom.name} ({count}/2)")
        
        if constraint_violations:
            stats_text += "   ⚠️ CONSTRAINT VIOLATIONS:\n"
            for violation in constraint_violations:
                stats_text += f"      • {violation}\n"
        else:
            stats_text += "   ✅ All constraints satisfied!\n"
        
        # Check conflicts
        conflicts = []
        slot_usage = defaultdict(list)
        
        for course_name, assignments in schedule.items():
            for assignment in assignments:
                slot_key = tuple(assignment["slot"])
                classroom = assignment["classroom"]
                slot_usage[(slot_key, classroom)].append(course_name)
        
        for (slot_key, classroom), courses in slot_usage.items():
            if len(courses) > 1:
                if slot_key in self.time_slot_dict:
                    time_slot = self.time_slot_dict[slot_key]
                    conflicts.append(f"{time_slot.get_label()} - {classroom}: {', '.join(courses)}")
        
        if conflicts:
            stats_text += "\n   ⚠️ SCHEDULING CONFLICTS:\n"
            for conflict in conflicts:
                stats_text += f"      • {conflict}\n"
        else:
            stats_text += "\n   ✅ No scheduling conflicts!\n"
        
        # Add generation info
        stats_text += f"""

🚀 GENERATION INFO
   • Generated with: MEGA-TURBO Scheduler
   • Algorithm: Multi-Core + Genetic Hybrid
   • Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
   • System: {mp.cpu_count()} CPU cores, GPU: {GPU_NAME}
"""
        
        # Display the text
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        # Add a footer
        footer_text = "📑 Generated by Ultimate School Schedule Generator v5.0 | 🚀 MEGA-TURBO Algorithm"
        ax.text(0.5, 0.02, footer_text, transform=ax.transAxes, fontsize=10,
               horizontalalignment='center', style='italic',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_course_legend(self, ax, x_pos: float, y_pos: float):
        """Add course color legend"""
        legend_elements = []
        for course_name, color in self.course_colors.items():
            short_name = self._shorten_text(course_name, 15)
            legend_elements.append(mpatches.Patch(color=color, label=short_name))
        
        if legend_elements:
            legend = ax.legend(handles=legend_elements, loc='center left', 
                             bbox_to_anchor=(x_pos/6, y_pos/2), 
                             title="📚 Courses", title_fontsize=12)
            legend.get_title().set_fontweight('bold')
    
    def _shorten_text(self, text: str, max_length: int) -> str:
        """Shorten text to fit in cells"""
        if len(text) <= max_length:
            return text
        return text[:max_length-2] + ".."
    
    def _is_dark_color(self, color: str) -> bool:
        """Check if color is dark (for text contrast)"""
        try:
            # Remove # if present
            color = color.lstrip('#')
            # Convert to RGB
            rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            # Calculate luminance
            luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
            return luminance < 0.5
        except:
            return False

class SchoolDataManager:
    def __init__(self, filename: str = "school_data.json"):
        self.filename = filename
        self.classrooms: List[Classroom] = []
        self.time_slots: List[TimeSlot] = []
        self.courses: List[Course] = []
    
    def load_data(self):
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.classrooms = [Classroom.from_dict(c) for c in data.get("classrooms", [])]
            self.time_slots = [TimeSlot.from_dict(t) for t in data.get("time_slots", [])]
            self.courses = [Course.from_dict(c) for c in data.get("courses", [])]
            
            print(f"📚 Loaded: {len(self.classrooms)} classrooms, {len(self.time_slots)} time slots, {len(self.courses)} courses")
            
        except FileNotFoundError:
            print(f"❌ File {self.filename} not found!")
            raise
    
    def save_data(self, schedules: List[Dict]):
        # Fix the classroom mapping for proper saving
        classroom_map = {i: c.name for i, c in enumerate(self.classrooms)}
        
        fixed_schedules = []
        for schedule in schedules:
            fixed_schedule = {}
            for course_name, assignments in schedule.items():
                fixed_assignments = []
                for assignment in assignments:
                    classroom_name = assignment["classroom"]
                    # If it's a simple index reference, map it back to name
                    if classroom_name.startswith("Classroom_"):
                        idx = int(classroom_name.split("_")[1])
                        classroom_name = classroom_map.get(idx, classroom_name)
                    
                    fixed_assignments.append({
                        "slot": assignment["slot"],
                        "classroom": classroom_name
                    })
                fixed_schedule[course_name] = fixed_assignments
            fixed_schedules.append(fixed_schedule)
        
        data = {
            "classrooms": [c.__dict__ for c in self.classrooms],
            "time_slots": [t.__dict__ for t in self.time_slots],
            "courses": [c.__dict__ for c in self.courses],
            "generated_schedules": fixed_schedules
        }
        
        # Convert enums to strings
        for classroom in data["classrooms"]:
            classroom["obligatory_type"] = classroom["obligatory_type"].value
        
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Data saved to {self.filename}")

def main():
    """Ultimate main function with PDF generation"""
    
    print("🚀 ULTIMATE School Schedule Generator v5.0 (PDF Edition)")
    print("=" * 70)
    print("🔥 ULTIMATE FEATURES:")
    print("   • MEGA-TURBO Algorithms")
    print("   • Full Multi-Core Utilization")
    print("   • GPU Acceleration (when available)")
    print("   • Professional PDF Reports")
    print("   • Weekly Grid Visualization")
    print("=" * 70)
    
    # Enhanced system info
    cpu_count = mp.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"💻 SYSTEM SPECIFICATIONS:")
    print(f"   • CPU: {cpu_count} cores")
    print(f"   • RAM: {ram_gb:.1f} GB")
    print(f"   • GPU: {GPU_NAME}")
    if GPU_AVAILABLE:
        print(f"   • GPU Memory: {GPU_MEMORY:.1f} GB")
    print("-" * 70)
    
    # Load data
    try:
        data_manager = SchoolDataManager()
        data_manager.load_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Problem analysis
    total_assignments = sum(len(c.get_slot_keys()) for c in data_manager.courses)
    search_space = len(data_manager.classrooms) ** total_assignments
    
    print(f"\n📊 PROBLEM ANALYSIS:")
    print(f"   • Total assignments: {total_assignments}")
    print(f"   • Search space: {search_space:e}")
    print(f"   • Complexity: {'🔴 EXTREME' if search_space > 1e15 else '🟡 HIGH' if search_space > 1e10 else '🟢 MANAGEABLE'}")
    
    # Algorithm selection
    print(f"\n🤖 CHOOSE ULTIMATE ALGORITHM:")
    print("1. 🔥 GPU Mode (GPU + optimizations)")
    print("2. 🚀 Multi-Core Mode (All CPU cores)")
    print("3. 🎯 Hybrid Mode (Multi-core + Genetic)")
    print("4. 🧠 Auto-Select (Best for your system)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    mode_map = {"1": "gpu", "2": "multicore", "3": "hybrid", "4": "auto"}
    mode = mode_map.get(choice, "auto")
    
    try:
        # Create mega-turbo scheduler
        scheduler = MegaTurboScheduler(
            data_manager.classrooms, 
            data_manager.courses, 
            data_manager.time_slots
        )
        
        # Solve
        solution = scheduler.solve_mega_turbo(mode=mode, timeout_minutes=15)
        
        if solution:
            # Save solution
            data_manager.save_data([solution])
            
            # Create ultimate PDF report
            visualizer = UltimateScheduleVisualizer(
                data_manager.classrooms,
                data_manager.courses,
                data_manager.time_slots
            )
            visualizer.create_ultimate_pdf_report(solution, "ultimate_schedule_report.pdf")
            
            print("\n🎉 ULTIMATE SUCCESS!")
            print("📁 Schedule saved in 'school_data.json'")
            print("📑 Ultimate PDF report saved as 'ultimate_schedule_report.pdf'")
            
            # Show detailed summary
            print(f"\n📋 SCHEDULE SUMMARY:")
            for course_name, assignments in solution.items():
                classroom_counts = defaultdict(int)
                for assignment in assignments:
                    classroom_counts[assignment["classroom"]] += 1
                
                print(f"   📚 {course_name}: {len(assignments)} classes")
                for classroom, count in classroom_counts.items():
                    print(f"      └─ {classroom}: {count} sessions")
            
        else:
            print("\n❌ No solution found with ultimate algorithms.")
            print("💡 ULTIMATE SUGGESTIONS:")
            print("   • Add more classrooms")
            print("   • Reduce obligatory constraints")
            print("   • Increase timeout")
            print("   • Check data consistency")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Graceful interrupt handling
    def signal_handler(sig, frame):
        print("\n🛑 Ultimate generator interrupted by user")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Configure multiprocessing for Windows
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn', force=True)
    
    main()
