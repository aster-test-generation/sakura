from.base import BaseSearcher
from sakura.nl2test.preprocessing.vector_stores import ClassVectorStore
class ClassSearcher(BaseSearcher[ClassVectorStore]):
	def _doc_to_result(B,doc):A='declaring_class_name';return{A:doc.metadata[A]}