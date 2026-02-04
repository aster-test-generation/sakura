'\nExceptions Package\n'
_A=None
class CustomBaseException(Exception):
	'Base exception'
	def __init__(A,message:str)->_A:super().__init__(message)
class ConfigurationException(CustomBaseException):
	'Error thrown when configuration parameters are not set or when agent policy is violated.'
	def __init__(C,config_var,message=_A,*,details=_A)->_A:
		B=message;A=config_var
		if not B:B=f"The configuration parameter {A} has not been set. Please set the configuration parameter {A} before proceeding."
		super().__init__(B);C.config_var=A;C.details=details