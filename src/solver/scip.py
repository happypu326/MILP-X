import copy
import enum
from typing import Optional, Any
import sys

import numpy as np
import pandas as pd
import pyscipopt as scip
from pyscipopt import SCIP_EVENTTYPE, SCIP_PARAMSETTING
import time

from src.solver.base_solver import Solver


class SCIPSolver(Solver):
	def __str__(self):
		return 'scip'

	def __init__(self):
		self.logger = None
		self.mps_path = None
		self.model = scip.Model('SCIP Model')

	@staticmethod
	def varname(var):
		return var.name

	def varval(self, var):
		return self.model.getVal(var)

	def get_vars(self):
		return self.model.getVars()

	def load_model(self, mip_path: str):
		self.model.readProblem(mip_path)
		self.mps_path = mip_path

	def hide_output_to_console(self):
		self.model.hideOutput(True)

	def create_integer_var(self, name, lower_bound=None, upper_bound=None):
		v = self.model.addVar(name=name, lb=lower_bound, ub=upper_bound, vtype="I")
		return v

	def create_real_var(self, name, lower_bound=None, upper_bound=None):
		v = self.model.addVar(name=name, lb=lower_bound, ub=upper_bound, vtype="C")
		return v

	def create_binary_var(self, name):
		v = self.model.addVar(name=name, vtype="B")
		return v

	def set_objective_function(self, equation, maximize=True):
		self.model.setObjective(equation)
		if maximize:
			self.model.setMaximize()
		else:
			self.model.setMinimize()

	def add_constraint(self, cns, name):
		self.model.addCons(cns, name)

	def solve(self, means='scip', log_file='', time_limit=3600, threads=12) -> pd.DataFrame:
		if threads == 0:
			threads = 1000
		print(f'Solving {self.mps_path} with {means}...')
		if log_file != '':
			with open(log_file, 'wb'):
				pass
		self.logger = SCIPLogger(means)
		self.model.setLogfile(log_file)
		self.model.setParam("limits/time", time_limit)
		self.model.setParam("parallel/maxnthreads", threads)
		self.model.setParam('randomization/randomseedshift', 0)
		self.model.setParam('randomization/lpseed', 0)
		self.model.setParam('randomization/permutationseed', 0)
		self.model.includeEventhdlr(self.logger, desc='get node info', name='SCIPLogger')
		self.model.optimize()
		print('Done solving with ' + means)

		self.logger.logs['Means'] = pd.Series(np.repeat(means, len(self.logger.logs)), index=self.logger.logs.index)
		self.logger.logs['MpsPath'] = pd.Series(np.repeat(self.mps_path, len(self.logger.logs)), index=self.logger.logs.index)

		return self.logger.logs

	def get_best_solution(self):
		sol = None
		if self.model.getNSols() > 0:
			sol = self.model.getBestSol()
		return sol

	def disable_presolver(self):
		self.model.setPresolve(SCIP_PARAMSETTING.OFF)
		self.model.setBoolParam("lp/presolving", False)

	def disable_cuts(self):
		self.model.setSeparating(SCIP_PARAMSETTING.OFF)

	def disable_heuristics(self):
		self.model.setHeuristics(SCIP_PARAMSETTING.OFF)

	def set_aggressive(self):
		self.model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)

	def get_sol_data(self):
		sols = []
		objs = []

		for sol in self.model.getSols():
			sols.append(np.array([sol[var] for var in self.get_vars()]))
			objs.append(self.model.getSolObjVal(sol))

		sols = np.array(sols, dtype=np.float32)
		objs = np.array(objs, dtype=np.float32)
		return sols, objs

	def collect_early_solution(self, time_limit=20, threads=1, top_k=3):
		"""Probing solve for EnCore (see GurobiSolver.collect_early_solution).
		Returns (early_sols, x_ES, t_probe)."""
		if threads == 0:
			threads = 1
		self.model.setParam("limits/time", time_limit)
		self.model.setParam("parallel/maxnthreads", threads)
		start = time.time()
		self.model.optimize()
		t_probe = time.time() - start

		mvars = self.get_vars()
		scip_sols = self.model.getSols()
		if not scip_sols:
			return [], None, t_probe
		sols, objs = [], []
		for sol in scip_sols:
			sols.append(np.array([sol[v] for v in mvars], dtype=np.float32))
			objs.append(float(self.model.getSolObjVal(sol)))
		maximize = (self.model.getObjectiveSense() == "maximize")
		order = np.argsort(objs)
		if maximize:
			order = order[::-1]
		ordered = [sols[i] for i in order]
		return ordered[:top_k], ordered[0], t_probe



class SCIPLogger(scip.Eventhdlr):

	def eventinit(self):
		self.logs = pd.DataFrame([{'Gap': 1, 'Time': 0.0, 'Means': 'scip'}])
		self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
		self.start_time = time.monotonic()

	def eventexec(self, event):
		end_time = time.monotonic()
		obj = self.model.getSolObjVal(self.model.getBestSol())
		log_entry = dict()
		log_entry['Incumbent'] = obj
		log_entry['Time'] = end_time - self.start_time + self.logs.loc[len(self.logs) - 1, 'Time']
		log_entry['Gap'] = self.model.getGap() / 100

		if end_time == self.start_time:
			self.logs.iloc[-1] = pd.DataFrame([log_entry])
		else:
			self.logs = pd.concat([self.logs, pd.DataFrame([log_entry])], ignore_index=True)
		self.start_time = end_time
