import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # 向量数据库配置
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
    
    # 文档处理配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # 检索配置
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    
    # 数据目录
    DATA_DIRS = {
        'papers': './data/papers',
        'tools_docs': './data/tools_docs', 
        'project_docs': './data/project_docs',
        'examples': './data/examples'
    }
    
    # 支持的文件类型
    SUPPORTED_EXTENSIONS = ['.txt', '.md', '.pdf', '.html', '.py', '.js', '.java', '.cpp', '.c']
