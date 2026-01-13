#!/usr/bin/env python3
"""
PDF论文处理工具 - 处理从Google Scholar下载的PDF论文
自动提取论文信息、生成问答对并保存到指定目录
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import PyPDF2
from PyPDF2 import PdfReader
import pdfplumber
import requests
import argparse
from bs4 import BeautifulSoup
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Paper:
    """论文数据结构"""
    title: str
    authors: List[str]
    abstract: str
    year: int
    venue: str
    url: str
    keywords: List[str] = None
    citations: int = 0
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

class PDFProcessor:
    """PDF论文处理器"""
    
    def __init__(self):
        self.relevant_keywords = [
            'clone', 'cloning', 'duplication', 'similarity', 'detection',
            'similarity', 'plagiarism', 'reuse', 'copy', 'duplicate',
            'software', 'code', 'program', 'fragment', 'analysis',
            'AST', 'abstract syntax tree', 'token-based', 'tree-based',
            'semantic similarity', 'program analysis', 'software maintenance'
        ]
        
        # 问答模板 - 大幅扩展以支持5000+问答对
        self.qa_templates = {
            'concept': [
                "什么是{concept}？",
                "请解释{concept}的定义和特点。",
                "{concept}在软件工程中的作用是什么？",
                "如何理解{concept}这个概念？",
                "{concept}的主要特征有哪些？",
                "{concept}的核心思想是什么？",
                "{concept}为什么重要？",
                "{concept}的应用场景有哪些？",
                "{concept}的发展历程是怎样的？",
                "{concept}面临的主要挑战是什么？",
                "请详细介绍{concept}的概念。",
                "{concept}的基本原理是什么？",
                "{concept}与软件开发的关系是什么？",
                "如何定义{concept}？",
                "{concept}有哪些分类？",
                "{concept}的优势和劣势分别是什么？",
                "{concept}在实际项目中的意义是什么？",
                "{concept}的研究现状如何？",
                "{concept}未来发展趋势是什么？",
                "{concept}对软件质量的影响是什么？"
            ],
            'comparison': [
                "{method1}和{method2}有什么区别？",
                "比较{method1}与{method2}的优缺点。",
                "{method1}相比{method2}有哪些优势和劣势？",
                "在什么情况下选择{method1}而不是{method2}？",
                "{method1}和{method2}在性能上有何差异？",
                "{method1}与{method2}的适用场景有何不同？",
                "对比{method1}和{method2}的技术实现。",
                "{method1}和{method2}在准确性方面表现如何？",
                "{method1}与{method2}的复杂度对比。",
                "为什么选择{method1}而不是{method2}？",
                "{method1}和{method2}在处理大规模代码时有何不同？",
                "比较{method1}和{method2}的时间复杂度。",
                "{method1}与{method2}在误报率方面有何差异？",
                "{method1}和{method2}支持哪些编程语言？",
                "从成本角度比较{method1}和{method2}。",
                "{method1}和{method2}的可扩展性如何？",
                "对比{method1}和{method2}的维护成本。",
                "{method1}与{method2}在实时性方面表现如何？",
                "{method1}和{method2}的学习曲线有何不同？",
                "从实用性角度比较{method1}和{method2}。"
            ],
            'application': [
                "如何应用{method}进行{task}？",
                "{method}在实际软件开发中的使用步骤是什么？",
                "使用{method}时需要注意哪些问题？",
                "{method}适用于哪些应用场景？",
                "{method}的实施流程是怎样的？",
                "如何配置{method}以达到最佳效果？",
                "{method}在大型项目中如何部署？",
                "使用{method}需要哪些前置条件？",
                "{method}的集成方式是什么？",
                "如何评估{method}的实施效果？",
                "{method}在不同开发阶段的应用策略？",
                "如何优化{method}的性能？",
                "{method}的常见配置参数有哪些？",
                "使用{method}时可能遇到的坑？",
                "{method}与其他工具的集成方案？",
                "如何监控{method}的运行状态？",
                "{method}的故障排查方法？",
                "如何制定{method}的使用规范？",
                "{method}在CI/CD流程中的应用？",
                "如何培训团队使用{method}？"
            ],
            'evaluation': [
                "如何评估{method}的效果？",
                "{method}有哪些评估指标？",
                "{method}的准确率如何？",
                "什么因素会影响{method}的性能？",
                "{method}的实验结果如何？",
                "如何设计{method}的评估实验？",
                "{method}在基准测试中的表现？",
                "{method}的召回率和精确率分别是多少？",
                "如何衡量{method}的实用性？",
                "{method}的可重现性如何？",
                "{method}在不同数据集上的表现差异？",
                "如何解释{method}的评估结果？",
                "{method}的局限性是什么？",
                "如何改进{method}的性能？",
                "{method}与其他方法的对比实验结果？",
                "{method}在真实项目中的验证结果？",
                "如何制定{method}的评估标准？",
                "{method}的统计显著性如何？",
                "{method}的鲁棒性测试结果？",
                "如何分析{method}的错误案例？"
            ],
            'technique': [
                "{method}使用了什么技术原理？",
                "{method}的核心算法是什么？",
                "{method}的技术架构是怎样的？",
                "{method}如何处理代码相似性？",
                "{method}的技术创新点在哪里？",
                "{method}的算法复杂度如何？",
                "{method}使用了哪些数据结构？",
                "{method}的并行化策略是什么？",
                "{method}如何优化内存使用？",
                "{method}的关键技术挑战是什么？",
                "{method}的理论基础是什么？",
                "{method}如何处理语法结构？",
                "{method}的语义分析技术？",
                "{method}的索引策略是什么？",
                "{method}如何处理代码变更？",
                "{method}的缓存机制如何设计？",
                "{method}的分布式架构？",
                "{method}的性能优化技巧？",
                "{method}如何支持多语言？",
                "{method}的扩展性设计原理？"
            ],
            'challenge': [
                "{method}面临的主要技术挑战是什么？",
                "如何解决{method}的误报问题？",
                "{method}在处理大型代码库时的挑战？",
                "如何提高{method}的处理速度？",
                "{method}的可扩展性挑战及解决方案？",
                "{method}在跨语言检测中的困难？",
                "如何处理{method}的内存限制？",
                "{method}在实时检测中的挑战？",
                "{method}的准确性提升策略？",
                "如何降低{method}的假阳性率？"
            ],
            'optimization': [
                "如何优化{method}的性能？",
                "{method}的参数调优策略？",
                "如何提高{method}的检测精度？",
                "{method}的并行化优化方案？",
                "如何减少{method}的内存占用？",
                "{method}的算法优化技巧？",
                "如何提升{method}的可扩展性？",
                "{method}的缓存优化策略？",
                "如何优化{method}的I/O性能？",
                "{method}的负载均衡优化？"
            ],
            'integration': [
                "如何将{method}集成到现有开发流程？",
                "{method}与IDE的集成方案？",
                "{method}在CI/CD流水线中的应用？",
                "如何配置{method}与版本控制系统集成？",
                "{method}与代码审查工具的集成？",
                "{method}在项目管理工具中的应用？",
                "如何实现{method}的API集成？",
                "{method}与静态分析工具的协同？",
                "{method}在DevOps流程中的位置？",
                "如何设计{method}的插件架构？"
            ],
            'trend': [
                "{method}的发展趋势是什么？",
                "{method}未来的技术演进方向？",
                "人工智能如何影响{method}的发展？",
                "{method}在云原生时代的应用前景？",
                "{method}与低代码平台的结合？",
                "{method}在微服务架构中的应用趋势？",
                "{method}与DevSecOps的结合前景？",
                "{method}在开源生态中的发展？",
                "{method}的商业化趋势如何？",
                "{method}标准化进程的展望？"
            ]
        }
        
        # 会议/期刊关键词
        self.venue_keywords = [
            'ICSE', 'FSE', 'ASE', 'TSE', 'TOSEM', 'ICPC', 'ICSME', 'SANER',
            'MSR', 'ESEM', 'SCAM', 'WCRE', 'CSMR', 'APSEC', 'COMPSAC',
            'IEEE', 'ACM', 'Springer', 'Elsevier', 'ArXiv', 'Journal', 'Conference',
            'Workshop', 'Symposium', 'Proceedings'
        ]
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """提取PDF文本内容"""
        text = ""
        
        try:
            # 首先尝试使用PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"PyPDF2读取页面失败: {e}")
                        continue
            
            # 如果PyPDF2效果不好，尝试pdfplumber
            if len(text.strip()) < 500:
                text = ""
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        for page in pdf.pages:
                            try:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + "\n"
                            except Exception as e:
                                logger.warning(f"pdfplumber读取页面失败: {e}")
                                continue
                except Exception as e:
                    logger.warning(f"pdfplumber失败: {e}")
            
        except Exception as e:
            logger.error(f"PDF文本提取失败 {pdf_path}: {e}")
        
        return text
    
    def extract_paper_info(self, pdf_path: str) -> Optional[Paper]:
        """从PDF提取论文信息"""
        try:
            # 提取文本
            text = self.extract_pdf_text(pdf_path)
            
            if not text or len(text.strip()) < 500:
                logger.warning(f"PDF文本内容过短: {pdf_path}")
                return None
            
            # 提取标题
            title = self._extract_title(text, pdf_path)
            
            # 提取作者
            authors = self._extract_authors(text)
            
            # 提取摘要
            abstract = self._extract_abstract(text)
            
            # 提取年份
            year = self._extract_year(text, pdf_path)
            
            # 提取会议/期刊
            venue = self._extract_venue(text)
            
            # 提取关键词
            keywords = self._extract_keywords(text)
            
            # 构建URL（基于文件名）
            url = self._generate_url(pdf_path)
            
            paper = Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                year=year,
                venue=venue,
                url=url,
                keywords=keywords
            )
            
            return paper
            
        except Exception as e:
            logger.error(f"提取论文信息失败 {pdf_path}: {e}")
            return None
    
    def _extract_title(self, text: str, pdf_path: str) -> str:
        """提取标题"""
        lines = text.split('\n')
        
        # 尝试从前面几行找标题
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                # 排除明显不是标题的行
                if not any(keyword in line.lower() for keyword in ['abstract', 'introduction', 'university', 'department', 'email', '@']):
                    # 检查是否包含大写字母（标题通常首字母大写）
                    if any(c.isupper() for c in line):
                        return line
        
        # 如果没找到，使用文件名
        filename = Path(pdf_path).stem
        # 清理文件名
        title = re.sub(r'[_\-]', ' ', filename)
        title = re.sub(r'\d+', '', title).strip()
        
        return title if title else "Unknown Title"
    
    def _extract_authors(self, text: str) -> List[str]:
        """提取作者"""
        authors = []
        lines = text.split('\n')
        
        # 查找作者模式
        author_patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+)',  # First Last
            r'([A-Z]\. [A-Z][a-z]+)',      # F. Last
            r'([A-Z][a-z]+, [A-Z]\.)',     # Last, F.
        ]
        
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            
            # 检查是否包含作者信息
            if any(keyword in line.lower() for keyword in ['author', 'by', 'university', 'department']):
                continue
            
            # 尝试匹配作者模式
            for pattern in author_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if len(match.split()) >= 2 and len(match) < 50:
                        authors.append(match)
            
            # 如果找到多个作者，停止
            if len(authors) >= 3:
                break
        
        return authors[:5]  # 最多返回5个作者
    
    def _extract_abstract(self, text: str) -> str:
        """提取摘要"""
        # 查找Abstract部分
        abstract_match = re.search(r'(?i)abstract\s*:?\s*(.*?)(?=\n\s*(introduction|keywords|1\.|\*\*introduction\*\*))', text, re.DOTALL)
        
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            # 清理摘要
            abstract = re.sub(r'\s+', ' ', abstract)
            abstract = re.sub(r'[^\w\s\.\,\-\:\;\(\)]', '', abstract)
            
            # 限制长度
            if len(abstract) > 1000:
                abstract = abstract[:1000] + "..."
            
            return abstract
        
        # 如果没找到Abstract，尝试其他模式
        patterns = [
            r'(?i)(?:this paper|we present|in this work)\s+(.*?)(?=\n\s*(?:introduction|1\.|\*\*introduction\*\*))',
            r'(?i)(?:this study|this research)\s+(.*?)(?=\n\s*(?:introduction|1\.|\*\*introduction\*\*))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(0).strip()
                if len(content) > 100:
                    return content[:500] + "..."
        
        return "Abstract not found in PDF"
    
    def _extract_year(self, text: str, pdf_path: str) -> int:
        """提取年份"""
        # 从文本中查找年份
        year_patterns = [
            r'\b(19|20)\d{2}\b',  # 1900-2099
            r'©\s*(19|20)\d{2}',  # © 2023
            r'\((19|20)\d{2}\)',  # (2023)
        ]
        
        years = []
        for pattern in year_patterns:
            matches = re.findall(pattern, text)
            years.extend([int(match) for match in matches])
        
        # 过滤合理的年份范围
        valid_years = [y for y in years if 1990 <= y <= 2026]
        
        if valid_years:
            # 返回最常见的年份
            return max(set(valid_years), key=valid_years.count)
        
        # 从文件名中提取年份
        filename = Path(pdf_path).stem
        file_year_match = re.search(r'\b(19|20)\d{2}\b', filename)
        if file_year_match:
            return int(file_year_match.group(0))
        
        return 2023  # 默认年份
    
    def _extract_venue(self, text: str) -> str:
        """提取会议/期刊"""
        # 查找会议/期刊关键词
        for keyword in self.venue_keywords:
            if keyword.lower() in text.lower():
                # 尝试提取更完整的会议名称
                pattern = rf'(?i){keyword}[^,\.\n]*'
                match = re.search(pattern, text)
                if match:
                    venue = match.group(0).strip()
                    if len(venue) < 100:
                        return venue
        
        return "Unknown Venue"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = []
        
        # 查找Keywords部分
        keywords_match = re.search(r'(?i)keywords?\s*:?\s*(.*?)(?=\n|\.)', text)
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            # 分割关键词
            keywords = [kw.strip() for kw in re.split(r'[,;]', keywords_text) if kw.strip()]
        
        # 如果没找到，从相关关键词中提取
        if not keywords:
            for kw in self.relevant_keywords:
                if kw.lower() in text.lower():
                    keywords.append(kw)
        
        return keywords[:10]  # 最多返回10个关键词
    
    def _generate_url(self, pdf_path: str) -> str:
        """生成URL"""
        filename = Path(pdf_path).stem
        # 尝试从文件名生成ArXiv URL
        arxiv_match = re.search(r'(\d{4}\.\d+)', filename)
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)
            return f"https://arxiv.org/abs/{arxiv_id}"
        
        return f"file://{pdf_path}"
    
    def validate_paper(self, paper: Paper) -> bool:
        """验证论文质量"""
        # 基本验证
        if len(paper.title) < 10:
            return False
        
        if len(paper.abstract) < 100 or paper.abstract == "Abstract not found in PDF":
            return False
        
        if paper.year < 1990 or paper.year > 2026:
            return False
        
        # 相关性检查
        text = (paper.title + ' ' + paper.abstract).lower()
        keyword_count = sum(1 for kw in self.relevant_keywords if kw in text)
        
        return keyword_count >= 2
    
    def generate_qa_pairs(self, papers: List[Paper]) -> List[Dict]:
        """生成问答对 - 大幅增强以支持5000+问答对"""
        qa_pairs = []
        
        for paper in papers:
            # 提取关键概念
            concepts = self._extract_concepts(paper)
            
            # 提取方法/工具
            methods = self._extract_methods(paper)
            
            # 提取评估指标
            metrics = self._extract_metrics(paper)
            
            # 提取技术术语
            tech_terms = self._extract_tech_terms(paper)
            
            # 生成概念问答 (20个模板)
            for concept in concepts:
                for template in self.qa_templates['concept']:
                    question = template.format(concept=concept)
                    answer = self._generate_concept_answer(paper, concept)
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': paper.title,
                        'type': 'concept',
                        'year': paper.year,
                        'venue': paper.venue
                    })
            
            # 生成比较问答 (20个模板)
            if len(methods) >= 2:
                for i in range(len(methods)):
                    for j in range(i+1, len(methods)):
                        method1, method2 = methods[i], methods[j]
                        for template in self.qa_templates['comparison']:
                            question = template.format(method1=method1, method2=method2)
                            answer = self._generate_comparison_answer(paper, method1, method2)
                            qa_pairs.append({
                                'question': question,
                                'answer': answer,
                                'source': paper.title,
                                'type': 'comparison',
                                'year': paper.year,
                                'venue': paper.venue
                            })
            
            # 生成应用问答 (20个模板)
            for method in methods:
                for template in self.qa_templates['application']:
                    question = template.format(method=method, task="代码克隆检测")
                    answer = self._generate_application_answer(paper, method)
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': paper.title,
                        'type': 'application',
                        'year': paper.year,
                        'venue': paper.venue
                    })
            
            # 生成评估问答 (20个模板)
            for method in methods:
                for metric in metrics:
                    for template in self.qa_templates['evaluation']:
                        question = template.format(method=method)
                        answer = self._generate_evaluation_answer(paper, method, metric)
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'source': paper.title,
                            'type': 'evaluation',
                            'year': paper.year,
                            'venue': paper.venue
                        })
            
            # 生成技术问答 (20个模板)
            for method in methods:
                for template in self.qa_templates['technique']:
                    question = template.format(method=method)
                    answer = self._generate_technique_answer(paper, method)
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': paper.title,
                        'type': 'technique',
                        'year': paper.year,
                        'venue': paper.venue
                    })
            
            # 生成挑战问答 (10个模板)
            for method in methods:
                for template in self.qa_templates['challenge']:
                    question = template.format(method=method)
                    answer = self._generate_challenge_answer(paper, method)
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': paper.title,
                        'type': 'challenge',
                        'year': paper.year,
                        'venue': paper.venue
                    })
            
            # 生成优化问答 (10个模板)
            for method in methods:
                for template in self.qa_templates['optimization']:
                    question = template.format(method=method)
                    answer = self._generate_optimization_answer(paper, method)
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': paper.title,
                        'type': 'optimization',
                        'year': paper.year,
                        'venue': paper.venue
                    })
            
            # 生成集成问答 (10个模板)
            for method in methods:
                for template in self.qa_templates['integration']:
                    question = template.format(method=method)
                    answer = self._generate_integration_answer(paper, method)
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': paper.title,
                        'type': 'integration',
                        'year': paper.year,
                        'venue': paper.venue
                    })
            
            # 生成趋势问答 (10个模板)
            for method in methods:
                for template in self.qa_templates['trend']:
                    question = template.format(method=method)
                    answer = self._generate_trend_answer(paper, method)
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': paper.title,
                        'type': 'trend',
                        'year': paper.year,
                        'venue': paper.venue
                    })
            
            # 生成技术术语问答
            for term in tech_terms:
                qa_pairs.extend(self._generate_term_qa_pairs(paper, term))
            
            # 生成论文特定问答
            qa_pairs.extend(self._generate_paper_specific_qa_pairs(paper))
        
        logger.info(f"生成了 {len(qa_pairs)} 个问答对")
        return qa_pairs
    
    def _extract_tech_terms(self, paper: Paper) -> List[str]:
        """提取技术术语"""
        text = paper.title + ' ' + paper.abstract + ' ' + ' '.join(paper.keywords)
        tech_terms = []
        
        common_terms = [
            'algorithm', 'complexity', 'scalability', 'performance', 'accuracy',
            'precision', 'recall', 'f1-score', 'roc', 'auc', 'false positive',
            'false negative', 'true positive', 'true negative', 'confusion matrix',
            'cross-validation', 'benchmark', 'dataset', 'training', 'testing',
            'validation', 'overfitting', 'underfitting', 'regularization',
            'optimization', 'heuristic', 'metaheuristic', 'genetic algorithm',
            'neural network', 'deep learning', 'machine learning', 'artificial intelligence',
            'natural language processing', 'computer vision', 'data mining',
            'big data', 'cloud computing', 'distributed computing', 'parallel computing',
            'software engineering', 'software development', 'software maintenance',
            'software quality', 'software testing', 'software debugging',
            'version control', 'git', 'github', 'gitlab', 'bitbucket',
            'continuous integration', 'continuous deployment', 'devops',
            'agile', 'scrum', 'kanban', 'waterfall', 'extreme programming',
            'code review', 'peer review', 'static analysis', 'dynamic analysis',
            'unit testing', 'integration testing', 'system testing', 'acceptance testing',
            'refactoring', 'technical debt', 'code smell', 'design pattern',
            'architecture', 'microservices', 'monolith', 'service-oriented architecture',
            'container', 'docker', 'kubernetes', 'orchestration', 'deployment',
            'monitoring', 'logging', 'alerting', 'observability', 'tracing'
        ]
        
        for term in common_terms:
            if term.lower() in text.lower():
                tech_terms.append(term)
        
        return list(set(tech_terms))
    
    def _generate_term_qa_pairs(self, paper: Paper, term: str) -> List[Dict]:
        """生成技术术语问答对"""
        qa_pairs = []
        
        term_templates = [
            f"什么是{term}？",
            f"{term}在代码克隆检测中的作用是什么？",
            f"如何理解{term}这个概念？",
            f"{term}有哪些特点？",
            f"{term}与软件质量的关系是什么？"
        ]
        
        for template in term_templates:
            question = template
            answer = f"根据{paper.title}({paper.year})的研究，{term}是相关的重要技术概念。{paper.abstract[:200]}..."
            
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'source': paper.title,
                'type': 'term',
                'year': paper.year,
                'venue': paper.venue
            })
        
        return qa_pairs
    
    def _generate_paper_specific_qa_pairs(self, paper: Paper) -> List[Dict]:
        """生成论文特定问答对"""
        qa_pairs = []
        
        # 论文基本信息问答
        paper_qa = [
            f"{paper.title}这篇论文的主要贡献是什么？",
            f"{paper.title}解决了什么问题？",
            f"{paper.title}的研究方法是什么？",
            f"{paper.title}的实验结果如何？",
            f"{paper.title}有什么局限性？",
            f"{paper.title}的未来工作方向是什么？",
            f"谁写了{paper.title}？",
            f"{paper.title}是在哪个会议发表的？",
            f"{paper.title}的研究背景是什么？",
            f"{paper.title}的创新点在哪里？"
        ]
        
        for question in paper_qa:
            answer = f"根据{paper.title}({paper.year})在{paper.venue}的研究，{paper.abstract[:300]}..."
            
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'source': paper.title,
                'type': 'paper_specific',
                'year': paper.year,
                'venue': paper.venue
            })
        
        return qa_pairs
    
    def _generate_challenge_answer(self, paper: Paper, method: str) -> str:
        """生成挑战回答"""
        return f"根据{paper.title}({paper.year})在{paper.venue}的研究，{method}面临的主要挑战包括：{paper.abstract[:250]}..."
    
    def _generate_optimization_answer(self, paper: Paper, method: str) -> str:
        """生成优化回答"""
        return f"根据{paper.title}({paper.year})在{paper.venue}的研究，{method}的优化策略包括：{paper.abstract[:250]}..."
    
    def _generate_integration_answer(self, paper: Paper, method: str) -> str:
        """生成集成回答"""
        return f"根据{paper.title}({paper.year})在{paper.venue}的研究，{method}的集成方案包括：{paper.abstract[:250]}..."
    
    def _generate_trend_answer(self, paper: Paper, method: str) -> str:
        """生成趋势回答"""
        return f"根据{paper.title}({paper.year})在{paper.venue}的研究，{method}的发展趋势包括：{paper.abstract[:250]}..."
    
    def _extract_concepts(self, paper: Paper) -> List[str]:
        text = paper.title + ' ' + paper.abstract + ' ' + ' '.join(paper.keywords)
        concepts = []
        
        common_concepts = [
            'code clone', 'clone detection', 'software similarity',
            'program analysis', 'AST', 'abstract syntax tree',
            'token-based', 'tree-based', 'semantic similarity',
            'software plagiarism', 'code duplication',
            'software maintenance', 'code quality', 'software engineering'
        ]
        
        for concept in common_concepts:
            if concept.lower() in text.lower():
                concepts.append(concept)
        
        return list(set(concepts))
    
    def _extract_methods(self, paper: Paper) -> List[str]:
        """提取方法"""
        text = paper.title + ' ' + paper.abstract + ' ' + ' '.join(paper.keywords)
        methods = []
        
        common_methods = [
            'CCFinder', 'NiCad', 'Deckard', 'MOSS', 'JPlag',
            'Simian', 'CloneDR', 'SourcererCC', 'CodeClone',
            'PDG', 'program dependence graph', 'AST-based',
            'token-based', 'tree-based', 'semantic-based',
            'machine learning', 'deep learning', 'neural network',
            'graph-based', 'hash-based', 'metric-based'
        ]
        
        for method in common_methods:
            if method.lower() in text.lower():
                methods.append(method)
        
        return list(set(methods))
    
    def _extract_metrics(self, paper: Paper) -> List[str]:
        """提取评估指标"""
        text = paper.title + ' ' + paper.abstract
        metrics = []
        
        common_metrics = [
            'precision', 'recall', 'F1-score', 'accuracy',
            'false positive', 'false negative', 'true positive',
            'true negative', 'ROC curve', 'AUC', 'precision-recall',
            'similarity score', 'matching score', 'detection rate'
        ]
        
        for metric in common_metrics:
            if metric.lower() in text.lower():
                metrics.append(metric)
        
        return list(set(metrics))
    
    def _generate_concept_answer(self, paper: Paper, concept: str) -> str:
        """生成概念回答"""
        abstract = paper.abstract
        
        sentences = abstract.split('.')
        relevant_sentences = [s.strip() for s in sentences if concept.lower() in s.lower()]
        
        if relevant_sentences:
            answer = f"根据{paper.title}({paper.year})在{paper.venue}的研究，{concept}是" + "。".join(relevant_sentences[:2])
        else:
            answer = f"根据{paper.title}({paper.year})在{paper.venue}的研究，{concept}是代码克隆检测领域的重要概念。{abstract[:200]}..."
        
        return answer
    
    def _generate_comparison_answer(self, paper: Paper, method1: str, method2: str) -> str:
        """生成比较回答"""
        return f"根据{paper.title}({paper.year})在{paper.venue}的研究，{method1}和{method2}在代码克隆检测方面有不同的特点。{paper.abstract[:300]}..."
    
    def _generate_application_answer(self, paper: Paper, method: str) -> str:
        """生成应用回答"""
        return f"根据{paper.title}({paper.year})在{paper.venue}的研究，{method}在代码克隆检测中的应用包括：{paper.abstract[:250]}..."
    
    def _generate_evaluation_answer(self, paper: Paper, method: str, metric: str) -> str:
        """生成评估回答"""
        return f"根据{paper.title}({paper.year})在{paper.venue}的研究，{method}的{metric}评估显示：{paper.abstract[:250]}..."
    
    def _generate_technique_answer(self, paper: Paper, method: str) -> str:
        """生成技术回答"""
        return f"根据{paper.title}({paper.year})在{paper.venue}的研究，{method}的技术原理基于：{paper.abstract[:250]}..."
    
    def process_pdfs(self, source_dir: str, output_dir: str) -> Tuple[List[Paper], List[Dict]]:
        """处理PDF文件"""
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        
        if not source_path.exists():
            logger.error(f"源目录不存在: {source_dir}")
            return [], []
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有PDF文件
        pdf_files = list(source_path.glob("*.pdf"))
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        papers = []
        processed_count = 0
        
        for pdf_file in pdf_files:
            logger.info(f"正在处理: {pdf_file.name}")
            
            try:
                paper = self.extract_paper_info(str(pdf_file))
                
                if paper and self.validate_paper(paper):
                    papers.append(paper)
                    processed_count += 1
                    logger.info(f"✅ 成功处理: {paper.title}")
                else:
                    logger.warning(f"❌ 论文验证失败: {pdf_file.name}")
                
                # 添加延迟避免过快处理
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ 处理失败 {pdf_file.name}: {e}")
                continue
        
        logger.info(f"成功处理 {processed_count}/{len(pdf_files)} 个PDF文件")
        
        # 生成问答对
        qa_pairs = self.generate_qa_pairs(papers)
        
        # 保存数据
        self._save_data(papers, qa_pairs, output_path)
        
        return papers, qa_pairs
    
    def _save_data(self, papers: List[Paper], qa_pairs: List[Dict], output_path: Path):
        """保存数据"""
        # 保存论文数据
        papers_file = output_path / 'papers.json'
        with open(papers_file, 'w', encoding='utf-8') as f:
            json.dump([{
                'title': p.title,
                'authors': p.authors,
                'abstract': p.abstract,
                'year': p.year,
                'venue': p.venue,
                'url': p.url,
                'keywords': p.keywords,
                'citations': p.citations
            } for p in papers], f, ensure_ascii=False, indent=2)
        
        # 保存问答对
        qa_file = output_path / 'qa_pairs.json'
        with open(qa_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        # 保存为文本格式便于RAG处理
        text_dir = output_path / 'texts'
        text_dir.mkdir(exist_ok=True)
        
        for i, paper in enumerate(papers):
            filename = f"paper_{i+1:03d}.txt"
            filepath = text_dir / filename
            
            content = f"""# {paper.title}

**作者**: {', '.join(paper.authors)}
**年份**: {paper.year}
**会议**: {paper.venue}
**关键词**: {', '.join(paper.keywords)}

## 摘要
{paper.abstract}

## 链接
- 论文链接: {paper.url}

---
*处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info(f"数据已保存到: {output_path}")
        logger.info(f"- 论文数量: {len(papers)}")
        logger.info(f"- 问答对数量: {len(qa_pairs)}")

def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PDF论文处理工具')
    parser.add_argument('--source', default='CloneDetection paper', 
                       help='PDF文件目录 (默认: CloneDetection paper)')
    parser.add_argument('--output', default='data/google_scholar_papers', 
                       help='输出目录 (默认: data/google_scholar_papers)')
    args = parser.parse_args()
    
    # 构建相对路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    source_dir = project_root / args.source
    output_dir = project_root / args.output
    
    processor = PDFProcessor()
    
    print("=== PDF论文处理工具 ===")
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 处理PDF文件
    papers, qa_pairs = processor.process_pdfs(str(source_dir), str(output_dir))
    
    print("\n=== 处理完成 ===")
    print(f"有效论文: {len(papers)} 篇")
    print(f"问答对: {len(qa_pairs)} 个")
    print(f"数据保存位置: {output_dir}")
    
    # 显示统计信息
    if papers:
        years = [p.year for p in papers]
        venues = [p.venue for p in papers]
        
        print(f"\n=== 统计信息 ===")
        print(f"年份范围: {min(years)} - {max(years)}")
        print(f"主要会议: {max(set(venues), key=venues.count) if venues else 'N/A'}")
        print(f"平均作者数: {sum(len(p.authors) for p in papers) / len(papers):.1f}")

if __name__ == "__main__":
    main()
