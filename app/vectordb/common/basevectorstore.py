from typing import List, Dict, Any

class BaseVectorStore:
    def set_index(self, index_name: str):
        raise NotImplementedError
    def create_index(self, index_name: str, dimension: int, recreate: bool = False):
        raise NotImplementedError
    def ingest(self, corpus: List[str], dimension: int):
        raise NotImplementedError
    def hybrid_search(self, query: str, dimension:int, top_k: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError