import abc

import pandas as pd
import sys

class Solver(abc.ABC):

	@staticmethod
	def varname(var):
		raise NotImplementedError('varname not implemented')

	def get_vars(self):
		raise NotImplementedError('get_vars not implemented')

	def create_integer_var(self, name, lower_bound, upper_bound):
		raise NotImplementedError('create_integer_var not implemented')

	def create_real_var(self, name, lower_bound, upper_bound):
		raise NotImplementedError('create_real_var not implemented')

	def create_binary_var(self, name):
		raise NotImplementedError('create_binary_var not implemented')

	def set_objective_function(self, equation, maximize=True):
		raise NotImplementedError('set_objective_function not implemented')

	def add_constraint(self, cns, name):
		raise NotImplementedError('add_constraint not implemented')

	def load_model(self, mip_path: str):
		raise NotImplementedError('load_model not implemented')

	def solve(self, means, log_file='', time_limit=3600, threads=0) -> pd.DataFrame:
		raise NotImplementedError('solve not implemented')

	def disable_presolver(self):
		raise NotImplementedError('disable_presolver not implemented')

	def disable_cuts(self):
		raise NotImplementedError('disable_cuts not implemented')


	def disable_heuristics(self):
		raise NotImplementedError('disable_heuristics not implemented')


	def set_aggressive(self):
		raise NotImplementedError('set_aggressive not implemented')


	def get_sol_data(self):
		raise NotImplementedError('get_sol_data not implemented')

	def hide_output_to_console(self):
		raise NotImplementedError('hide_output_to_console not implemented')
	def copy_model(self):
		raise NotImplementedError('copy_model not implemented')
	def get_var_by_name(self, name):
		raise NotImplementedError

	def write_model(self, path: str):
		raise NotImplementedError

	def get_obj_val(self):
		raise NotImplementedError

	def get_node_count(self):
		raise NotImplementedError
