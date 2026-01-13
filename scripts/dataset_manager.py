#!/usr/bin/env python3
"""
数据集管理工具 - 管理、验证和统计代码克隆检测数据集
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DatasetStats:
    """数据集统计信息"""
    total_documents: int
    total_chunks: int
    total_qa_pairs: int
    file_types: Dict[str, int]
    year_distribution: Dict[int, int]
    venue_distribution: Dict[str, int]
    avg_document_length: float
    last_updated: str

class DatasetValidator:
    """数据集验证器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
    
    def validate_dataset(self) -> Dict:
        """验证整个数据集"""
        logger.info("开始验证数据集...")
        
        results = {
            'papers': self._validate_papers(),
            'qa_pairs': self._validate_qa_pairs(),
            'tools_docs': self._validate_tools_docs(),
            'project_docs': self._validate_project_docs(),
            'examples': self._validate_examples(),
            'overall': {}
        }
        
        # 计算总体统计
        results['overall'] = self._calculate_overall_stats(results)
        
        logger.info("数据集验证完成")
        return results
    
    def _validate_papers(self) -> Dict:
        """验证论文数据"""
        papers_dir = self.data_dir / 'papers'
        processed_dir = self.data_dir / 'google_scholar_papers'
        
        papers = []
        errors = []
        
        # 验证原有论文目录
        if papers_dir.exists():
            papers.extend(self._validate_papers_directory(papers_dir, errors))
        
        # 验证处理后的论文目录
        if processed_dir.exists():
            papers.extend(self._validate_papers_directory(processed_dir, errors))
        
        if not papers_dir.exists() and not processed_dir.exists():
            return {'status': 'missing', 'count': 0, 'errors': []}
        
        return {
            'status': 'valid' if not errors else 'warnings',
            'count': len(papers),
            'errors': errors,
            'papers': papers
        }
    
    def _validate_papers_directory(self, papers_dir: Path, errors: List) -> List[Dict]:
        """验证特定论文目录"""
        papers = []
        
        for file_path in papers_dir.rglob('*.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content.strip()) < 100:
                    errors.append(f"文件过短: {file_path}")
                    continue
                
                papers.append({
                    'file': str(file_path),
                    'size': len(content),
                    'lines': len(content.split('\n'))
                })
                
            except Exception as e:
                errors.append(f"读取失败 {file_path}: {e}")
        
        return papers
    
    def _validate_qa_pairs(self) -> Dict:
        """验证问答对数据"""
        qa_files = list(self.data_dir.rglob('*qa*.json'))
        if not qa_files:
            return {'status': 'missing', 'count': 0, 'errors': []}
        
        qa_pairs = []
        errors = []
        
        for qa_file in qa_files:
            try:
                with open(qa_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for i, pair in enumerate(data):
                        if not isinstance(pair, dict):
                            errors.append(f"无效格式 {qa_file}:{i}")
                            continue
                        
                        question = pair.get('question', '')
                        answer = pair.get('answer', '')
                        
                        if len(question) < 10 or len(answer) < 20:
                            errors.append(f"问答过短 {qa_file}:{i}")
                            continue
                        
                        qa_pairs.append(pair)
                
            except Exception as e:
                errors.append(f"读取失败 {qa_file}: {e}")
        
        return {
            'status': 'valid' if not errors else 'warnings',
            'count': len(qa_pairs),
            'errors': errors,
            'qa_pairs': qa_pairs
        }
    
    def _validate_tools_docs(self) -> Dict:
        """验证工具文档"""
        return self._validate_document_type('tools_docs')
    
    def _validate_project_docs(self) -> Dict:
        """验证项目文档"""
        return self._validate_document_type('project_docs')
    
    def _validate_examples(self) -> Dict:
        """验证示例代码"""
        return self._validate_document_type('examples')
    
    def _validate_document_type(self, doc_type: str) -> Dict:
        """验证特定类型的文档"""
        doc_dir = self.data_dir / doc_type
        if not doc_dir.exists():
            return {'status': 'missing', 'count': 0, 'errors': []}
        
        documents = []
        errors = []
        
        for file_path in doc_dir.rglob('*'):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if len(content.strip()) < 50:
                        errors.append(f"文件过短: {file_path}")
                        continue
                    
                    documents.append({
                        'file': str(file_path),
                        'size': len(content),
                        'type': file_path.suffix
                    })
                    
                except Exception as e:
                    errors.append(f"读取失败 {file_path}: {e}")
        
        return {
            'status': 'valid' if not errors else 'warnings',
            'count': len(documents),
            'errors': errors,
            'documents': documents
        }
    
    def _calculate_overall_stats(self, results: Dict) -> Dict:
        """计算总体统计"""
        total_docs = 0
        total_size = 0
        total_errors = 0
        
        for category, data in results.items():
            if category == 'overall':
                continue
            
            if isinstance(data, dict) and 'count' in data:
                total_docs += data['count']
            if isinstance(data, dict) and 'errors' in data:
                total_errors += len(data['errors'])
        
        return {
            'total_documents': total_docs,
            'total_errors': total_errors,
            'health_score': max(0, 100 - (total_errors / max(total_docs, 1) * 100))
        }

class DatasetAnalyzer:
    """数据集分析器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def analyze_dataset(self) -> DatasetStats:
        """分析数据集"""
        logger.info("开始分析数据集...")
        
        # 收集所有文档
        all_documents = self._collect_all_documents()
        
        # 分析文档类型
        file_types = self._analyze_file_types(all_documents)
        
        # 分析年份分布（如果有论文数据）
        year_distribution = self._analyze_year_distribution()
        
        # 分析会议分布
        venue_distribution = self._analyze_venue_distribution()
        
        # 计算平均文档长度
        avg_length = self._calculate_avg_length(all_documents)
        
        # 统计问答对数量
        qa_count = self._count_qa_pairs()
        
        # 估算分块数量
        chunk_count = self._estimate_chunks(all_documents)
        
        stats = DatasetStats(
            total_documents=len(all_documents),
            total_chunks=chunk_count,
            total_qa_pairs=qa_count,
            file_types=file_types,
            year_distribution=year_distribution,
            venue_distribution=venue_distribution,
            avg_document_length=avg_length,
            last_updated=datetime.now().isoformat()
        )
        
        logger.info("数据集分析完成")
        return stats
    
    def _collect_all_documents(self) -> List[Dict]:
        """收集所有文档信息"""
        documents = []
        
        for file_path in self.data_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.py', '.java', '.cpp', '.js']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    documents.append({
                        'path': str(file_path),
                        'size': len(content),
                        'type': file_path.suffix,
                        'category': self._get_category(file_path)
                    })
                except Exception as e:
                    logger.warning(f"无法读取文件 {file_path}: {e}")
        
        return documents
    
    def _get_category(self, file_path: Path) -> str:
        """获取文件所属类别"""
        parts = file_path.parts
        if 'papers' in parts:
            return 'papers'
        elif 'tools_docs' in parts:
            return 'tools_docs'
        elif 'project_docs' in parts:
            return 'project_docs'
        elif 'examples' in parts:
            return 'examples'
        else:
            return 'other'
    
    def _analyze_file_types(self, documents: List[Dict]) -> Dict[str, int]:
        """分析文件类型分布"""
        file_types = {}
        for doc in documents:
            file_type = doc['type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
        return file_types
    
    def _analyze_year_distribution(self) -> Dict[int, int]:
        """分析年份分布"""
        year_dist = {}
        
        # 从原有论文数据中提取年份
        papers_dir = self.data_dir / 'papers'
        if papers_dir.exists():
            for file_path in papers_dir.rglob('*.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        papers = json.load(f)
                    
                    for paper in papers:
                        year = paper.get('year')
                        if isinstance(year, int):
                            year_dist[year] = year_dist.get(year, 0) + 1
                except Exception as e:
                    logger.warning(f"无法分析年份分布 {file_path}: {e}")
        
        # 从处理后的论文数据中提取年份
        processed_dir = self.data_dir / 'google_scholar_papers'
        if processed_dir.exists():
            for file_path in processed_dir.rglob('*.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        papers = json.load(f)
                    
                    for paper in papers:
                        year = paper.get('year')
                        if isinstance(year, int):
                            year_dist[year] = year_dist.get(year, 0) + 1
                except Exception as e:
                    logger.warning(f"无法分析年份分布 {file_path}: {e}")
        
        return year_dist
    
    def _analyze_venue_distribution(self) -> Dict[str, int]:
        """分析会议分布"""
        venue_dist = {}
        
        # 从原有论文数据中提取会议
        papers_dir = self.data_dir / 'papers'
        if papers_dir.exists():
            for file_path in papers_dir.rglob('*.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        papers = json.load(f)
                    
                    for paper in papers:
                        venue = paper.get('venue', 'Unknown')
                        venue_dist[venue] = venue_dist.get(venue, 0) + 1
                except Exception as e:
                    logger.warning(f"无法分析会议分布 {file_path}: {e}")
        
        # 从处理后的论文数据中提取会议
        processed_dir = self.data_dir / 'google_scholar_papers'
        if processed_dir.exists():
            for file_path in processed_dir.rglob('*.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        papers = json.load(f)
                    
                    for paper in papers:
                        venue = paper.get('venue', 'Unknown')
                        venue_dist[venue] = venue_dist.get(venue, 0) + 1
                except Exception as e:
                    logger.warning(f"无法分析会议分布 {file_path}: {e}")
        
        return venue_dist
    
    def _calculate_avg_length(self, documents: List[Dict]) -> float:
        """计算平均文档长度"""
        if not documents:
            return 0.0
        
        total_size = sum(doc['size'] for doc in documents)
        return total_size / len(documents)
    
    def _count_qa_pairs(self) -> int:
        """统计问答对数量"""
        qa_count = 0
        
        # 搜索所有问答对文件
        qa_files = []
        qa_files.extend(self.data_dir.rglob('*qa*.json'))
        
        # 特别搜索处理后的论文目录
        processed_dir = self.data_dir / 'google_scholar_papers'
        if processed_dir.exists():
            qa_files.extend(processed_dir.rglob('qa_pairs.json'))
        
        for file_path in qa_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    qa_count += len(data)
            except Exception as e:
                logger.warning(f"无法统计问答对 {file_path}: {e}")
        
        return qa_count
    
    def _estimate_chunks(self, documents: List[Dict]) -> int:
        """估算分块数量（基于1000字符分块）"""
        chunk_size = 1000
        total_chunks = 0
        
        for doc in documents:
            chunks = max(1, doc['size'] // chunk_size)
            total_chunks += chunks
        
        return total_chunks
    
    def generate_report(self, stats: DatasetStats) -> str:
        """生成分析报告"""
        report = f"""
# 数据集分析报告

## 基本信息
- **总文档数**: {stats.total_documents}
- **预估分块数**: {stats.total_chunks}
- **问答对数量**: {stats.total_qa_pairs}
- **平均文档长度**: {stats.avg_document_length:.1f} 字符
- **最后更新**: {stats.last_updated}

## 文件类型分布
"""
        
        for file_type, count in stats.file_types.items():
            report += f"- **{file_type}**: {count} 个文件\n"
        
        if stats.year_distribution:
            report += "\n## 年份分布\n"
            for year, count in sorted(stats.year_distribution.items()):
                report += f"- **{year}**: {count} 篇论文\n"
        
        if stats.venue_distribution:
            report += "\n## 会议分布\n"
            for venue, count in stats.venue_distribution.items():
                report += f"- **{venue}**: {count} 篇论文\n"
        
        return report

class DatasetManager:
    """数据集管理器主类"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.validator = DatasetValidator(data_dir)
        self.analyzer = DatasetAnalyzer(data_dir)
    
    def run_full_analysis(self) -> Dict:
        """运行完整的数据集分析"""
        logger.info("开始完整数据集分析...")
        
        # 验证数据集
        validation_results = self.validator.validate_dataset()
        
        # 分析数据集
        stats = self.analyzer.analyze_dataset()
        
        # 生成报告
        report = self.analyzer.generate_report(stats)
        
        # 保存结果
        self._save_results(validation_results, stats, report)
        
        return {
            'validation': validation_results,
            'stats': asdict(stats),
            'report': report
        }
    
    def _save_results(self, validation: Dict, stats: DatasetStats, report: str):
        """保存分析结果"""
        output_dir = self.data_dir / 'analysis'
        output_dir.mkdir(exist_ok=True)
        
        # 保存验证结果
        with open(output_dir / 'validation.json', 'w', encoding='utf-8') as f:
            json.dump(validation, f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        with open(output_dir / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(asdict(stats), f, ensure_ascii=False, indent=2)
        
        # 保存报告
        with open(output_dir / 'report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"分析结果已保存到 {output_dir}")

def main():
    """主函数"""
    manager = DatasetManager()
    results = manager.run_full_analysis()
    
    # 打印简要结果
    print("\n=== 数据集分析结果 ===")
    print(f"总文档数: {results['stats']['total_documents']}")
    print(f"预估分块数: {results['stats']['total_chunks']}")
    print(f"问答对数量: {results['stats']['total_qa_pairs']}")
    print(f"健康评分: {results['validation']['overall']['health_score']:.1f}")
    
    print("\n详细报告已保存到 data/analysis/ 目录")

if __name__ == "__main__":
    main()
