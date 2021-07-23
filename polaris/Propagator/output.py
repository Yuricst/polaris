"""
Propagation output class
"""


import numpy as np


class Propout:
	def __init__(self, xs, ys, zs, vxs, vys, vzs, times, state0, statef, stms, dstatef, eventStates=None, eventTimes=None):
		"""Propagation output class

		Args:
			xs
			ys
			zs
			vxs
			vys
			vzs
			times
			state0
			statef
			stms
			dstatef
			eventStates (lst):
			eventTimes (lst):
		"""
		self.xs  = xs
		self.ys  = ys
		self.zs  = zs 
		self.vxs = vxs
		self.vys = vys
		self.vzs = vzs
		self.times   = times
		self.state0  = state0
		self.statef  = statef
		self.dstatef = dstatef
		self.stms        = stms
		self.eventStates = eventStates
		self.eventTimes  = eventTimes


	def state_at_index(self, idx):
		"""Get state at index idx

		Args:
			idx (int): index at which state-vector is extracted
		"""
		return np.array([self.xs[idx], self.ys[idx], self.zs[idx], self.vxs[idx], self.vys[idx], self.vzs[idx]])