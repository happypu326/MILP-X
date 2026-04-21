import os
import pickle
from multiprocessing import Process, Queue

from src.solver.solver_utils import SOLVER_CLASSES
from src.utils.utils import get_diffilo_data_train


class DIFFILOPreprocessor:

    def __init__(self, solver_name='gurobi', max_time=1000,
                 max_solutions=500, threads=1, workers=16):
        if solver_name not in SOLVER_CLASSES:
            raise ValueError(f"Unknown solver '{solver_name}', "
                             f"choose from {list(SOLVER_CLASSES.keys())}")
        self.solver_class = SOLVER_CLASSES[solver_name]
        self.solver_name = solver_name
        self.max_time = max_time
        self.max_solutions = max_solutions
        self.threads = threads
        self.workers = workers

    @staticmethod
    def prepare_directories(output_root):
        dirs = {
            'solutions': os.path.join(output_root, 'solutions'),
            'logs': os.path.join(output_root, 'logs'),
            'tensors': os.path.join(output_root, 'tensors'),
            'samples': os.path.join(output_root, 'samples'),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        return dirs

    def solve_instance(self, filepath, log_dir, mode):
        solver = self.solver_class()
        solver.hide_output_to_console()
        solver.load_model(filepath)

        if mode == 'train':
            solver.set_search_mode(mode=2)
            solver.set_search_PoolSolutions(max_solution=self.max_solutions)
        else:
            solver.set_aggressive()

        log_path = os.path.join(log_dir, f'{os.path.basename(filepath)}.log')
        solver.solve(log_file=log_path, time_limit=self.max_time,
                     threads=self.threads)

        variables = solver.get_vars()
        var_names = [solver.varname(var) for var in variables]
        solutions, objectives = solver.get_sol_data()

        return {
            'var_names': var_names,
            'sols': solutions,
            'objs': objectives,
        }

    def process_single_file(self, filepath, output_dirs, mode):
        solution_data = self.solve_instance(filepath, output_dirs['logs'], mode)

        graph, A, b, c = get_diffilo_data_train(filepath)
        tensor_data = (A, b, c)

        base_name = os.path.splitext(os.path.basename(filepath))[0]
        with open(os.path.join(output_dirs['solutions'], f'{base_name}.sol'), 'wb') as f:
            pickle.dump(solution_data, f)
        with open(os.path.join(output_dirs['tensors'], f'{base_name}.pkl'), 'wb') as f:
            pickle.dump(tensor_data, f)
        with open(os.path.join(output_dirs['samples'], f'{base_name}.pkl'), 'wb') as f:
            pickle.dump(graph, f) 
        
    def _worker(self, queue, input_dir, output_dirs, mode):
        while True:
            filename = queue.get()
            if filename is None:
                break
            filepath = os.path.join(input_dir, filename)
            try:
                self.process_single_file(filepath, output_dirs, mode)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    def process(self, input_dir, output_dir, mode='test'):
        os.makedirs(output_dir, exist_ok=True)
        output_dirs = self.prepare_directories(output_dir)

        existing = set(os.listdir(output_dirs['samples']))
        filenames = [f for f in os.listdir(input_dir)
                     if f'{os.path.splitext(f)[0]}.pkl' not in existing]

        if not filenames:
            print("All files have been processed, no new files to process.")
            return

        print(f"Files to process: {len(filenames)}, worker processes: {self.workers}")

        file_queue = Queue()
        for fn in filenames:
            file_queue.put(fn)
        for _ in range(self.workers):
            file_queue.put(None)

        processes = []
        for _ in range(self.workers):
            p = Process(target=self._worker,
                        args=(file_queue, input_dir, output_dirs, mode))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("Data processing completed.")
