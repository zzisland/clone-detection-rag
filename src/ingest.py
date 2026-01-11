import os
import re
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from bs4 import BeautifulSoup
import markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import Config

class DocumentProcessor:
    """文档处理器，负责读取和清洗各种格式的文档"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def read_file(self, file_path: str) -> str:
        """根据文件类型读取文件内容"""
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.txt':
                return self._read_txt(file_path)
            elif file_ext == '.md':
                return self._read_markdown(file_path)
            elif file_ext == '.pdf':
                return self._read_pdf(file_path)
            elif file_ext == '.html':
                return self._read_html(file_path)
            elif file_ext in ['.py', '.js', '.java', '.cpp', '.c']:
                return self._read_code(file_path)
            else:
                print(f"不支持的文件类型: {file_ext}")
                return ""
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return ""
    
    def _read_txt(self, file_path: str) -> str:
        """读取文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _read_markdown(self, file_path: str) -> str:
        """读取Markdown文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 转换为HTML然后提取纯文本
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
    
    def _read_pdf(self, file_path: str) -> str:
        """读取PDF文件"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _read_html(self, file_path: str) -> str:
        """读取HTML文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            soup = BeautifulSoup(content, 'html.parser')
            return soup.get_text()
    
    def _read_code(self, file_path: str) -> str:
        """读取代码文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()[\]{}"\'-]', ' ', text)
        # 移除过短的行
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
        return '\n'.join(lines)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        return self.text_splitter.split_documents(documents)

class DataIngestor:
    """数据摄取器，负责处理整个数据摄取流程"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        # 使用 HuggingFace 的中文 Embedding 模型
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",  # 中文向量模型，轻量高效
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"Embedding 模型加载完成: BAAI/bge-small-zh-v1.5 (设备: {device})")
    
    def load_documents_from_directory(self, directory: str) -> List[Document]:
        """从目录加载所有文档"""
        documents = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            print(f"目录不存在: {directory}")
            return documents
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in Config.SUPPORTED_EXTENSIONS:
                print(f"正在处理文件: {file_path}")
                content = self.processor.read_file(str(file_path))
                if content:
                    cleaned_content = self.processor.clean_text(content)
                    metadata = {
                        'source': str(file_path),
                        'file_type': file_path.suffix.lower(),
                        'file_name': file_path.name,
                        'directory': file_path.parent.name
                    }
                    doc = Document(page_content=cleaned_content, metadata=metadata)
                    documents.append(doc)
        
        return documents
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """创建向量数据库"""
        if not documents:
            print("没有文档可以处理")
            return None
        
        print(f"正在处理 {len(documents)} 个文档...")
        split_docs = self.processor.split_documents(documents)
        print(f"分割后得到 {len(split_docs)} 个文档块")
        
        # 创建向量数据库
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIRECTORY
        )
        
        # 持久化
        vector_store.persist()
        print(f"向量数据库已保存到: {Config.CHROMA_PERSIST_DIRECTORY}")
        
        return vector_store
    
    def ingest_all_data(self) -> Chroma:
        """摄取所有数据目录中的文档"""
        all_documents = []
        
        for dir_name, dir_path in Config.DATA_DIRS.items():
            print(f"\n正在处理目录: {dir_name}")
            documents = self.load_documents_from_directory(dir_path)
            all_documents.extend(documents)
            print(f"从 {dir_name} 加载了 {len(documents)} 个文档")
        
        if all_documents:
            return self.create_vector_store(all_documents)
        else:
            print("没有找到任何文档")
            return None

def main():
    """主函数，用于测试数据摄取功能"""
    ingestor = DataIngestor()
    vector_store = ingestor.ingest_all_data()
    
    if vector_store:
        print("数据摄取完成!")
        # 测试检索
        query = "什么是代码克隆检测?"
        docs = vector_store.similarity_search(query, k=3)
        print(f"\n测试检索 '{query}' 的结果:")
        for i, doc in enumerate(docs, 1):
            print(f"\n文档 {i}:")
            print(f"来源: {doc.metadata.get('source', 'Unknown')}")
            print(f"内容: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()

