from.base import BaseSearcher
from sakura.nl2test.preprocessing.vector_stores import MethodVectorStore
class MethodSearcher(BaseSearcher[MethodVectorStore]):
	def _doc_to_result(E,doc):D='method_signature';C='containing_class_name';B='declaring_class_name';A=doc;return{B:A.metadata[B],C:A.metadata[C],D:A.metadata[D]}