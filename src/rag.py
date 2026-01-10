from typing import List, Dict, Any, Optional
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.schema import BaseOutputParser
from langchain.memory import ConversationBufferMemory
from retriever import RetrieverManager
from config import Config

class CloneDetectionOutputParser(BaseOutputParser):
    """专门用于克隆检测问答的输出解析器"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """解析LLM输出"""
        # 简单的文本解析，可以扩展为更复杂的结构化输出
        return {
            "answer": text.strip(),
            "confidence": "high" if "明确" in text or "肯定" in text else "medium"
        }
    
    def get_format_instructions(self) -> str:
        return "请直接回答问题，不需要特殊格式。"

class CloneDetectionRAG:
    """代码克隆检测RAG系统"""
    
    def __init__(self):
        self.retriever_manager = RetrieverManager()
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 初始化提示模板
        self._setup_prompts()
        
        # 加载向量数据库
        self.retriever_manager.load_vector_store()
    
    def _setup_prompts(self):
        """设置各种提示模板"""
        
        # 通用问答模板
        self.qa_template = """你是一个专业的代码克隆检测专家助手。你的任务是帮助开发者理解代码克隆检测的相关知识。

基于以下检索到的文档内容，回答用户的问题。如果文档中没有相关信息，请基于你的专业知识回答，并明确说明。

检索到的文档：
{context}

用户问题：{question}

请提供准确、详细、专业的回答。如果涉及技术概念，请给出清晰的解释和示例。

回答："""
        
        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "question"]
        )
        
        # 代码分析模板
        self.code_analysis_template = """作为一个代码克隆检测专家，请分析以下代码片段或问题。

相关文档内容：
{context}

用户输入：
{user_input}

请从克隆检测的角度进行分析，包括：
1. 可能的克隆类型
2. 检测方法建议
3. 注意事项
4. 相关工具推荐

分析结果："""
        
        self.code_analysis_prompt = PromptTemplate(
            template=self.code_analysis_template,
            input_variables=["context", "user_input"]
        )
        
        # 工具比较模板
        self.tool_comparison_template = """请比较不同的代码克隆检测工具。

相关文档：
{context}

用户询问：{question}

请提供详细的工具比较，包括：
- 各工具的特点和优势
- 适用场景
- 性能表现
- 使用难度

比较结果："""
        
        self.tool_comparison_prompt = PromptTemplate(
            template=self.tool_comparison_template,
            input_variables=["context", "question"]
        )
    
    def answer_question(
        self, 
        question: str, 
        search_type: str = "general",
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """回答用户问题"""
        
        # 检索相关文档
        docs = self.retriever_manager.search(question, search_type, filters)
        
        if not docs:
            return {
                "answer": "抱歉，我没有找到相关的文档来回答您的问题。不过我可以基于一般知识为您解答。",
                "sources": [],
                "confidence": "low"
            }
        
        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 选择合适的提示模板
        prompt = self.qa_prompt
        
        # 生成回答
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(context=context, question=question)
        
        # 提取来源信息
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        
        return {
            "answer": result,
            "sources": sources,
            "context_used": len(docs),
            "confidence": "high" if len(docs) >= 3 else "medium"
        }
    
    def analyze_code(self, code_snippet: str) -> Dict[str, Any]:
        """分析代码片段"""
        
        # 构建查询
        query = f"代码片段分析：{code_snippet[:200]}..."
        
        # 检索相关文档
        docs = self.retriever_manager.search(query, search_type="general")
        
        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 使用代码分析模板
        chain = LLMChain(llm=self.llm, prompt=self.code_analysis_prompt)
        result = chain.run(context=context, user_input=code_snippet)
        
        return {
            "answer": result,
            "sources": [doc.metadata.get("source", "Unknown") for doc in docs],
            "code_length": len(code_snippet),
            "confidence": "high" if len(docs) >= 3 else "medium"
        }
    
    def compare_tools(self, tool_names: List[str]) -> Dict[str, Any]:
        """比较克隆检测工具"""
        
        # 构建查询
        query = f"比较工具：{', '.join(tool_names)}"
        
        # 检索相关文档
        docs = self.retriever_manager.search(query, search_type="by_type", filters={"type": "tools"})
        
        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 使用工具比较模板
        chain = LLMChain(llm=self.llm, prompt=self.tool_comparison_prompt)
        result = chain.run(context=context, question=query)
        
        return {
            "answer": result,
            "sources": [doc.metadata.get("source", "Unknown") for doc in docs],
            "tools_compared": tool_names,
            "confidence": "high" if len(docs) >= 2 else "medium"
        }
    
    def explain_concept(self, concept: str) -> Dict[str, Any]:
        """解释概念"""
        
        query = f"解释概念：{concept}"
        
        # 优先从论文和项目文档中搜索
        docs = self.retriever_manager.search(query, search_type="general")
        
        if not docs:
            # 如果没有找到，尝试更广泛的搜索
            docs = self.retriever_manager.search(concept, search_type="general")
        
        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 构建概念解释的专门提示
        concept_template = """请详细解释以下代码克隆检测相关的概念。

相关文档：
{context}

需要解释的概念：{concept}

请提供：
1. 概念定义
2. 背景和动机
3. 主要特点
4. 应用场景
5. 相关研究或工具

解释："""
        
        concept_prompt = PromptTemplate(
            template=concept_template,
            input_variables=["context", "concept"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=concept_prompt)
        result = chain.run(context=context, concept=concept)
        
        return {
            "answer": result,
            "sources": [doc.metadata.get("source", "Unknown") for doc in docs],
            "concept": concept,
            "confidence": "high" if len(docs) >= 3 else "medium"
        }
    
    def get_chat_response(self, message: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """聊天接口"""
        
        # 分析消息类型
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in ["比较", "对比", "vs"]):
            # 工具比较
            tools = [word.strip() for word in message.split() if word.strip()[0].isupper()]
            if tools:
                return self.compare_tools(tools)
        
        if any(keyword in message_lower for keyword in ["解释", "什么是", "定义"]):
            # 概念解释
            concept = message.replace("解释", "").replace("什么是", "").replace("定义", "").strip()
            if concept:
                return self.explain_concept(concept)
        
        if any(keyword in message_lower for keyword in ["代码", "函数", "类"]):
            # 代码分析
            return self.analyze_code(message)
        
        # 默认为通用问答
        return self.answer_question(message)

def main():
    """测试RAG功能"""
    rag = CloneDetectionRAG()
    
    # 测试不同类型的问题
    test_questions = [
        "什么是Type-1、Type-2、Type-3代码克隆？",
        "比较NiCad和CCFinder工具",
        "AST方法和Token方法有什么区别？",
        "如何评估克隆检测工具的性能？"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        result = rag.get_chat_response(question)
        print(f"回答: {result['answer'][:300]}...")
        print(f"来源数量: {len(result['sources'])}")

if __name__ == "__main__":
    main()
