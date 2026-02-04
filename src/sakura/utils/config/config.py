'\nConfig Module\n'
_C=False
_B=True
_A=None
import json,os,re
from typing import Any
from pathlib import Path
import toml
from sakura.utils.exceptions import ConfigurationException
from sakura.utils.llm.model import Provider
def init_config(project_name:str,*,base_project_dir:str,project_output_dir:str,use_stored_index:bool=_B,llm_model:str,llm_provider:Provider=_A,emb_provider:Provider=_A,emb_model:str=_A,llm_api_url:str=_A,emb_api_url:str=_A,llm_api_key:str=_A,emb_api_key:str=_A,localization_max_iters:int=40,composition_max_iters:int=40,supervisor_max_iters:int=5,can_parallel_tool:bool=_B,reuse_config:bool=_C,max_tokens:int=16384,configure_reasoning:bool=_C,reasoning_effort:str='medium',exclude_reasoning:bool=_B,openrouter_ignore_providers:list[str]|_A=_A,store_code_iteration:bool=_C)->'Config':
	T='composition';S='api_key';R='https://ete-litellm.bx.cloud9.ibm.com';Q='http://localhost:8000/v1';P='https://openrouter.ai/api/v1';O='https://api.openai.com/v1';N='model';M=openrouter_ignore_providers;L=emb_api_url;K=llm_api_url;J='project';I='max_iters';H='reasoning';G='provider';F=emb_provider;E=llm_provider;D='emb';C='api_url';B='llm';A=Config(_A,reuse=reuse_config);U=.4;V=.5;W=.3;X=.3
	if E is not _A:A.set(B,G,E.value)
	else:A.set(B,G,_A)
	if F is not _A:A.set(D,G,F.value)
	else:A.set(D,G,_A)
	A.set(B,N,llm_model);A.set(B,'can_parallel_tool',can_parallel_tool);A.set(B,'summarization_temp',U);A.set(B,'code_gen_temp',V);A.set(B,'decision_temp',W);A.set(B,'structured_temp',X);A.set(B,'max_tokens',max_tokens);A.set(H,'configure',configure_reasoning);A.set(H,'effort',reasoning_effort);A.set(H,'exclude',exclude_reasoning)
	if M:A.set('openrouter','ignore_providers',M)
	A.set(D,N,emb_model)
	if K is not _A:A.set(B,C,val=K)
	elif E==Provider.OPENAI:A.set(B,C,val=O)
	elif E==Provider.OPENROUTER:A.set(B,C,val=P)
	elif E==Provider.OLLAMA:A.set(B,C,val='http://localhost:11434/v1')
	elif E==Provider.VLLM:A.set(B,C,val=Q)
	elif E==Provider.GCP:A.set(B,C,val=R)
	elif E==Provider.MISTRAL:A.set(B,C,val='https://api.mistral.ai/v1')
	A.set(B,S,val=llm_api_key)
	if L is not _A:A.set(D,C,val=L)
	elif F==Provider.OPENAI:A.set(D,C,val=O)
	elif F==Provider.OPENROUTER:A.set(D,C,val=P)
	elif F==Provider.VLLM:A.set(D,C,val=Q)
	elif F==Provider.GCP:A.set(D,C,val=R)
	A.set(D,S,val=emb_api_key);A.set('localization',I,val=localization_max_iters);A.set(T,I,val=composition_max_iters);A.set('supervisor',I,val=supervisor_max_iters);A.set(J,'base_project_dir',val=base_project_dir);A.set(J,'project_output_dir',val=project_output_dir);A.set(J,'use_stored_index',val=use_stored_index);A.set(T,'store_code_iteration',val=store_code_iteration);return A
class Config:
	"This is singleton class to hold the configuration information.\n\n    By virtue of it's singleton nature, it can be configured once and used anywhere. Every time a new instance is created,\n    we have overridden the `__new__` magic method to return a pre-existing instance of this class.\n    ";_instance=_A;_LOCK_FILE=Path.cwd().joinpath('.aster.lock')
	def __new__(A,conf_file:Path=_A,reuse:bool=_B):
		'Create a new instance of the config class\n\n        Args:\n            conf_file (Path, optional): Path to the configuration file. Defaults to None.\n            reuse (bool, optional): True to reuse the lock file. Defaults to True.\n\n        Returns:\n            _type_: _description_\n        '
		if not A._instance:A._instance=super(Config,A).__new__(A);A._instance._conf_file=conf_file;A._instance._last_modified=_A;A._instance._load_config(reuse)
		return A._instance
	@classmethod
	def reset(A):
		'Reset the singleton instance to None'
		if A._instance:A._instance._conf_file=_A;A._instance._last_modified=_A;A._instance.config={};A._instance=_A
	@classmethod
	def destroy(A):
		'Remove the lock file and reset the singleton instance'
		if A._instance:
			if os.path.exists(A._LOCK_FILE):os.remove(A._LOCK_FILE)
			A.reset()
	def _load_config(A,reuse:bool):
		"Load the configurations from the TOML file. If the lock file exists, then load that.\n        If the file does not exist, return a FileNotFound error.\n        If a file wasn't specified, then return an empty dict.\n\n        Args:\n            reuse (bool): Reuse forces the use of the lock file.\n\n        Exceptions:\n            FileNotFound: Thrown if config file is not found\n        ";B=reuse;A.config={}
		if os.path.exists(A._LOCK_FILE)and B is _B:
			with open(A._LOCK_FILE,'r',encoding='utf8')as D:A.config=json.load(D);return
		if A._conf_file:
			try:
				C=os.path.getmtime(A._conf_file)
				if A._last_modified is _A or C!=A._last_modified or B is _C:A.config=toml.load(A._conf_file);A.last_modified=C;A._save_config_state()
			except FileNotFoundError as E:raise ConfigurationException('',message=f"Configuration file '{A._conf_file}' could not be found.")from E
	def _save_config_state(A):0
	def _expand_env_variables(B,value):
		'Expand environment variables in a given value.';A=value
		if isinstance(A,str):return re.sub('(?i)\\$(\\w+)|env:(\\w+)|\\$\\{(\\w+)\\}',lambda match:os.environ.get(match.group(1)or match.group(2)or match.group(3),match.group(0)),A)
		return A
	def get(B,section:str,key:str)->Any:
		'Get any value in a given section.\n\n        Args:\n            section (str): Configuration section.\n            key (str): Configuration key.\n\n        Returns:\n            Any: Value associated with that section.\n        ';C=key;A=section
		if A not in B.config:D=f'Group "{A}" is not found in config.';raise ConfigurationException('',message=D)
		if C not in B.config[A]:D=f'Parameter "{C}" in group "{A}" is not found in config.';raise ConfigurationException('',message=D)
		E=B._expand_env_variables(B.config[A][C]);return E
	def set(A,section:str,key:str,val:Any)->_A:
		'Set any value in a given section.\n\n        Args:\n            section (str): Configuration section.\n            key (str): Configuration key.\n            value (str): Configuration value to be set.\n        ';B=section
		if B not in A.config:A.config[B]={}
		A.config[B][key]=val;A._save_config_state()
	@property
	def LOCK_FILE(self):return self._LOCK_FILE