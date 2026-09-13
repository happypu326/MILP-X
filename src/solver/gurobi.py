import gurobipy
import numpy as np
import pandas as pd
import gurobipy as gb
from gurobipy import GRB
# NOTE: `gurobi_logtools` is only needed to parse solver logs inside solve();
# imported lazily there so merely importing this module (e.g. to build a model
# or predict) does not require the optional dependency.

from src.solver.base_solver import Solver
import sys
import time


class GurobiSolver(Solver):
	def __str__(self):
		return 'gurobi'

	def __init__(self):
		self.logs = None
		self.mps_path = None
		self.model: gurobipy.Model | None = None

	@staticmethod
	def varname(var):
		return var.VarName

	@staticmethod
	def varval(var):
		return var.X

	def get_vars(self):
		return self.model.getVars()

	def hide_output_to_console(self):
		gb.setParam('LogToConsole', 0)

	def create_integer_var(self, name, lower_bound=-GRB.INFINITY, upper_bound=GRB.INFINITY):
		v = self.model.addVar(name=name, lb=lower_bound, ub=upper_bound, vtype=GRB.INTEGER)
		return v

	def create_real_var(self, name, lower_bound=-GRB.INFINITY, upper_bound=GRB.INFINITY):
		v = self.model.addVar(name=name, lb=lower_bound, ub=upper_bound, vtype=GRB.CONTINUOUS)
		return v

	def create_binary_var(self, name):
		v = self.model.addVar(name=name, vtype=GRB.BINARY)
		return v

	def set_objective_function(self, equation, maximize=True):
		self.model.setObjective(equation)
		self.model.ModelSense = GRB.MAXIMIZE if maximize else GRB.MINIMIZE

	def add_constraint(self, cns, name):
		self.model.addConstr(cns, name)

	def load_model(self, mip_path: str):
		self.model = gb.read(mip_path)
		self.mps_path = mip_path

	def set_search_mode(self, mode=1):
		self.model.Params.PoolSearchMode = mode

	def set_search_PoolSolutions(self, max_solution=500):
		self.model.Params.PoolSolutions = max_solution

	def solve(self, means='gurobi', log_file='', time_limit=3600, threads=0) -> pd.DataFrame:
		print(f'Solving {self.mps_path} with {means}...')
		if log_file != '':
			with open(log_file, 'wb'):
				pass
		self.model.Params.LogFile = log_file
		self.model.Params.TimeLimit = time_limit
		self.model.Params.Threads = threads
		self.model.optimize()
		if log_file:
			import gurobi_logtools as glt
			self.logs = glt.parse(log_file).progress('nodelog')
			self.logs['Means'] = pd.Series(np.repeat(means, len(self.logs)), index=self.logs.index)
			self.logs['MpsPath'] = pd.Series(np.repeat(self.mps_path, len(self.logs)), index=self.logs.index)
		print('Done solving with ' + means)
		return self.logs

	def disable_presolver(self):
		self.model.Params.Presolve = 0

	def disable_cuts(self):
		self.model.Params.Cuts = 0

	def disable_heuristics(self):
		self.model.Params.Heuristics = 0

	def set_aggressive(self):
		self.model.Params.MIPFocus = 1

	def get_sol_data(self):
		sols = []
		objs = []
		solc = self.model.getAttr('SolCount')
		for sn in range(solc):
			self.model.Params.SolutionNumber = sn
			sols.append(np.array(self.model.Xn))
			objs.append(self.model.PoolObjVal)

		sols = np.array(sols, dtype=np.float32)
		objs = np.array(objs, dtype=np.float32)
		return sols, objs
	
	def copy_model(self):
		new_solver = GurobiSolver()
		new_solver.model = self.model.copy()
		new_solver.mps_path = self.mps_path
		return new_solver

	def get_var_by_name(self, name):
		return self.model.getVarByName(name)

	def write_model(self, path: str):
		self.model.write(path)

	def get_obj_val(self):
		if self.model.SolCount == 0:
			return None
		return float(self.model.ObjVal)

	def get_node_count(self):
		return self.model.NodeCount

	def get_status(self):
		return int(self.model.Status)
	
	def get_gap(self):
		if self.model.SolCount == 0:
			return None
		return float(self.model.MIPGap)

