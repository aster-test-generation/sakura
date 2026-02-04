from pathlib import Path
from langchain.schema import Document
from.faiss_base import BaseFAISSVectorStore
from sakura.nl2test.preprocessing.embedders import BaseEmbedder
from sakura.nl2test.models import ClassSnippet
from sakura.utils.config import Config
from sakura.utils.exceptions import ConfigurationException
def _class_to_doc(snippet:ClassSnippet)->Document:A=snippet;return Document(page_content=A.simple_class_name,metadata={'declaring_class_name':A.declaring_class_name})
class ClassVectorStore(BaseFAISSVectorStore[ClassSnippet]):
	def __init__(H,embedder:BaseEmbedder):
		F=False;E='project';C=None;D=Config()
		try:A=bool(D.get(E,'use_stored_index'))
		except ConfigurationException:A=F
		B:Path|C=C
		try:G=Path(D.get(E,'project_output_dir'));B=G/'indexes'/'class'
		except ConfigurationException:B=C;A=F
		super().__init__(embedder,_class_to_doc,index_dir=B,use_stored_index=A)