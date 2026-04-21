from .gurobi import GurobiSolver
from .scip import SCIPSolver
SOLVER_CLASSES = {'scip': SCIPSolver, 'gurobi' : GurobiSolver}