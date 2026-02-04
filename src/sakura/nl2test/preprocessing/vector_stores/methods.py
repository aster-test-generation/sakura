from pathlib import Path
from langchain.schema import Document
from.faiss_base import BaseFAISSVectorStore
from sakura.nl2test.preprocessing.embedders import BaseEmbedder
from sakura.nl2test.models import MethodSnippet
from sakura.utils.config import Config
from sakura.utils.exceptions import ConfigurationException
def _method_to_doc(snippet:MethodSnippet)->Document:A=snippet;return Document(page_content=A.code,metadata={'declaring_class_name':A.declaring_class_name,'containing_class_name':A.containing_class_name,'method_signature':A.method_signature})
class MethodVectorStore(BaseFAISSVectorStore[MethodSnippet]):
	def __init__(H,embedder:BaseEmbedder):
		F=False;E='project';C=None;D=Config()
		try:A=bool(D.get(E,'use_stored_index'))
		except ConfigurationException:A=F
		B:Path|C=C
		try:G=Path(D.get(E,'project_output_dir'));B=G/'indexes'/'method'
		except ConfigurationException:B=C;A=F
		super().__init__(embedder,_method_to_doc,index_dir=B,use_stored_index=A)