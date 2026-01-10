from typing import List, Dict, Any, Optional
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from config import Config

class CloneDetectionRetriever(BaseRetriever):
    """专门用于代码克隆检测的检索器"""
    
    def __init__(self, vector_store: Chroma, top_k: int = Config.TOP_K_RETRIEVAL):
        super().__init__()
        self.vector_store = vector_store
        self.top_k = top_k
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL
        )
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """检索相关文档"""
        # 基础相似性搜索
        docs = self.vector_store.similarity_search(query, k=self.top_k)
        
        # 可以在这里添加更复杂的检索逻辑
        # 例如：基于文档类型的过滤、重排序等
        
        return docs
    
    def search_with_metadata(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """带过滤条件的搜索"""
        if filters:
            # 构建过滤条件
            search_kwargs = {"k": self.top_k}
            if "file_type" in filters:
                search_kwargs["filter"] = {"file_type": filters["file_type"]}
            if "directory" in filters:
                search_kwargs["filter"] = {"directory": filters["directory"]}
            
            docs = self.vector_store.similarity_search(query, **search_kwargs)
        else:
            docs = self.vector_store.similarity_search(query, k=self.top_k)
        
        return docs
    
    def get_documents_by_type(self, query: str, doc_type: str) -> List[Document]:
        """根据文档类型检索"""
        type_mapping = {
            "papers": "papers",
            "tools": "tools_docs", 
            "project": "project_docs",
            "examples": "examples"
        }
        
        directory = type_mapping.get(doc_type.lower())
        if directory:
            return self.search_with_metadata(query, {"directory": directory})
        else:
            return self.search_with_metadata(query)

class RetrieverManager:
    """检索器管理器"""
    
    def __init__(self):
        self.vector_store = None
        self.retriever = None
    
    def load_vector_store(self) -> bool:
        """加载已存在的向量数据库"""
        try:
            self.vector_store = Chroma(
                persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
                embedding_function=OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    openai_api_key=Config.OPENAI_API_KEY,
                    openai_api_base=Config.OPENAI_BASE_URL
                )
            )
            self.retriever = CloneDetectionRetriever(self.vector_store)
            print("向量数据库加载成功")
            return True
        except Exception as e:
            print(f"加载向量数据库失败: {e}")
            return False
    
    def search(
        self, 
        query: str, 
        search_type: str = "general",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """统一的搜索接口"""
        if not self.retriever:
            if not self.load_vector_store():
                return []
        
        if search_type == "general":
            return self.retriever._get_relevant_documents(query, run_manager=None)
        elif search_type == "filtered":
            return self.retriever.search_with_metadata(query, filters)
        elif search_type == "by_type":
            doc_type = filters.get("type", "general") if filters else "general"
            return self.retriever.get_documents_by_type(query, doc_type)
        else:
            return self.retriever._get_relevant_documents(query, run_manager=None)
    
    def get_search_summary(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """获取搜索结果摘要"""
        if not docs:
            return {
                "query": query,
                "total_results": 0,
                "document_types": {},
                "sources": []
            }
        
        # 统计文档类型分布
        doc_types = {}
        sources = []
        
        for doc in docs:
            # 统计文件类型
            file_type = doc.metadata.get("file_type", "unknown")
            doc_types[file_type] = doc_types.get(file_type, 0) + 1
            
            # 收集来源
            source = doc.metadata.get("source", "unknown")
            if source not in sources:
                sources.append(source)
        
        return {
            "query": query,
            "total_results": len(docs),
            "document_types": doc_types,
            "sources": sources,
            "sample_content": docs[0].page_content[:200] + "..." if docs else ""
        }

def main():
    """测试检索功能"""
    manager = RetrieverManager()
    
    if manager.load_vector_store():
        # 测试不同类型的搜索
        test_queries = [
            "什么是代码克隆检测?",
            "AST方法和Token方法的区别",
            "如何评估克隆检测工具",
            "Type-1 Type-2 Type-3克隆"
        ]
        
        for query in test_queries:
            print(f"\n搜索: {query}")
            docs = manager.search(query)
            
            summary = manager.get_search_summary(query, docs)
            print(f"找到 {summary['total_results']} 个结果")
            print(f"文档类型分布: {summary['document_types']}")
            
            for i, doc in enumerate(docs[:3], 1):
                print(f"\n结果 {i}:")
                print(f"来源: {doc.metadata.get('source', 'Unknown')}")
                print(f"内容预览: {doc.page_content[:150]}...")

if __name__ == "__main__":
    main()
