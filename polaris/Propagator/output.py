"""
Propagation output class
"""


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