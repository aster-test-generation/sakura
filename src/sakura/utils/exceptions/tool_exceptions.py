from langchain_core.tools import ToolException
from typing import Dict,Any,Optional
from abc import ABC
class BaseToolException(ToolException,ABC):
	'Base class for tool-related exceptions.'
	def __init__(A,message:str,extra_info:Optional[Dict[str,Any]]=None)->None:super().__init__(message);A.extra_info=extra_info or{}
	def __str__(A)->str:B=super().__str__();return f"{B} Details: {A.extra_info}"if A.extra_info else B
class InvalidArgumentError(BaseToolException):'Raised when an invalid argument is provided to a tool.'
class MethodNotFoundError(BaseToolException):'Raised when a method cannot be found in the specified class.'
class ClassNotFoundError(BaseToolException):'Raised when a class cannot be found.'
class CallSiteNotFoundError(BaseToolException):'Raised when call sites cannot be found.'
class FormatError(BaseToolException):'Raised when output format validation fails for LLMs.'
class ClassFileNotFound(BaseToolException):'Raised when a file associated to a qualified class cannot be found.'
class CompilationUnitNotFound(BaseToolException):'Raised when a compilation unit cannot be found.'
class BlockNotFoundError(BaseToolException):'Raised when a block cannot be found.'
class FileDeletionError(BaseToolException):'Raised when a file deletion operation fails.'
class PomXmlNotFoundError(BaseToolException):"Raised when the project's root pom.xml is missing."
class ProjectCompilationError(BaseToolException):'Raised when unrelated project compilation errors block execution.'