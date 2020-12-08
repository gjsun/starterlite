import sys
import numpy as np
from .SetDefaultParameters import _cosmo_params, _hmf_params, _sensitivity_params, _cibmodel_params, _dust_params, \
                                  _grf_params, _wf_params, _ham_params

class ParameterFile(dict):
	def __init__(self, **kwargs):
		
		self.cosmo_params = _cosmo_params
		self.hmf_params = _hmf_params
		self.sensitivity_params = _sensitivity_params
		self.grf_params = _grf_params
		self.wf_params = _wf_params
		self.cibmodel_params = _cibmodel_params.copy()
		self.dust_params = _dust_params
		self.ham_params = _ham_params
		
		if sys.version_info[0] < 3:
			for key, value in kwargs.iteritems():

				if 'wf_' in key:
					if (key in _wf_params.keys()) and (value == _wf_params[key]):
						self.modifiedwfparams = False
					else:
						self.wf_params[key] = value
						self.modifiedwfparams = True

				if 'grf_' in key:
					if (key in _grf_params.keys()) and (value == _grf_params[key]):
						self.modifiedgrfparams = False
					else:
						self.grf_params[key] = value
						self.modifiedgrfparams = True

				if 'sens_' in key:
					if (key in _sensitivity_params.keys()) and (value == _sensitivity_params[key]):
						self.modifiedsensparams = False
					else:
						self.sensitivity_params[key] = value
						self.modifiedsensparams = True

				if 'cib_' in key:
					if (key in _cibmodel_params.keys()) and (value == _cibmodel_params[key]):
						self.modifiedcibparams = False
					else:
						self.cibmodel_params[key] = value
						self.modifiedcibparams = True

				if 'dust_' in key:
					if (key in _dust_params.keys()) and (value == _dust_params[key]):
						self.modifieddustparams = False
					else:
						self.dust_params[key] = value
						self.modifieddustparams = True

				if 'hmf_' in key:
					if (key in _hmf_params.keys()) and (value == _hmf_params[key]):
						self.modifiedhmfparams = False
					else:
						self.hmf_params[key] = value
						self.modifiedhmfparams = True
						
		else:
			for key, value in kwargs.items():

				if 'wf_' in key:
					if (key in _wf_params.keys()) and (value == _wf_params[key]):
						self.modifiedwfparams = False
					else:
						self.wf_params[key] = value
						self.modifiedwfparams = True

				if 'grf_' in key:
					if (key in _grf_params.keys()) and (value == _grf_params[key]):
						self.modifiedgrfparams = False
					else:
						self.grf_params[key] = value
						self.modifiedgrfparams = True

				if 'sens_' in key:
					if (key in _sensitivity_params.keys()) and (value == _sensitivity_params[key]):
						self.modifiedsensparams = False
					else:
						self.sensitivity_params[key] = value
						self.modifiedsensparams = True

				if 'cib_' in key:
					if (key in _cibmodel_params.keys()) and (value == _cibmodel_params[key]):
						self.modifiedcibparams = False
					else:
						self.cibmodel_params[key] = value
						self.modifiedcibparams = True

				if 'dust_' in key:
					if (key in _dust_params.keys()) and (value == _dust_params[key]):
						self.modifieddustparams = False
					else:
						self.dust_params[key] = value
						self.modifieddustparams = True

				if 'hmf_' in key:
					if (key in _hmf_params.keys()) and (value == _hmf_params[key]):
						self.modifiedhmfparams = False
					else:
						self.hmf_params[key] = value
						self.modifiedhmfparams = True