'\nRichLog is a thin wrapper around the Python logging module that\nadds convenient helpers and a Rich console handler by default.\n\nNotes\n- By default, logs go to the console only (via Rich).\n- Use add_file_handler() to also write logs to a file.\n'
_C='markup'
_B=None
_A=True
import logging
from rich.logging import RichHandler
from pathlib import Path
FORMAT='%(message)s'
logging.basicConfig(level='INFO',format=FORMAT,datefmt='[%X]',handlers=[RichHandler(show_path=False)])
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('langchain_openai').setLevel(logging.WARNING)
class RichLog:
	'Common Logger that uses Rich';log=logging.getLogger('rich')
	@staticmethod
	def info(msg:str):'Log level INFO';RichLog.log.info(msg,extra={_C:_A})
	@staticmethod
	def warn(msg:str):'Log level WARNING';RichLog.log.warning(msg,extra={_C:_A})
	@staticmethod
	def debug(msg:str):'Log level DEBUG';RichLog.log.debug(msg,extra={_C:_A})
	@staticmethod
	def error(msg:str):'Log level ERROR';RichLog.log.error(msg,extra={_C:_A})
	@staticmethod
	def activate_debug():'Sets the logging level to DEBUG';RichLog.log.setLevel(logging.DEBUG)
	@staticmethod
	def set_level(level:int|str)->_B:'Set log level on both this logger and the root logger.\n\n        This helps when third-party libraries emit useful debug logs.\n        ';A=level;logging.getLogger().setLevel(A);RichLog.log.setLevel(A)
	@staticmethod
	def add_file_handler(file_path:str,*,overwrite:bool=False,level:int|str|_B=_B)->_B:
		'Add a file handler so logs are also written to disk.\n\n        Parameters\n        - file_path: destination log file path (created if missing).\n        - overwrite: when True, truncates the file; otherwise appends.\n        - level: optional log level for this handler (defaults to logger level).\n        ';C=level;B=file_path
		try:Path(B).parent.mkdir(parents=_A,exist_ok=_A)
		except Exception:pass
		D='w'if overwrite else'a';A=logging.FileHandler(B,mode=D,encoding='utf-8');A.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
		if C is not _B:A.setLevel(C)
		RichLog.log.addHandler(A)