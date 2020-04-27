import rpy2.robjects as robjects

class RWinOutWatcher(object):

	def __init__(self, ip):
		self.shell = ip
		self.original_run_cell = ip.run_cell
		self.printROut = False


	def post_execute(self):
		if(self.printROut):
			self.printROut = False
			__RROUT__ = robjects.r['..RROUT..']
			for line in __RROUT__:
				print(line)


	def run_cell(self, raw_cell, **kw):
		if ( raw_cell.strip().startswith('%%R') ):
			
			codeIdx = raw_cell.find('\n')+1
			header = raw_cell[0:codeIdx]
			Rcode = raw_cell[codeIdx:]
			
			# compute output just when there  are lines of code (header and code)
			if codeIdx > 0 and len(Rcode) > 0:
				self.printROut = True
				return self.original_run_cell(header + '\n..RROUT.. <- captureOutput({\n' + Rcode + '\n})', **kw)
			
		# otherwise, use original method
		return self.original_run_cell(raw_cell, **kw)



def load_ipython_extension(ip):
	rw = RWinOutWatcher(ip)
	
	# loading magic and libraries
	ip.run_line_magic('load_ext', 'rpy2.ipython')
	ip.run_line_magic('R', 'library(R.utils)')
	ip.run_line_magic('config', 'Application.verbose_crash=True')
	
	# registering events
	ip.events.register('post_execute', rw.post_execute)
	
	# dirty hack
	ip.run_cell = rw.run_cell
	