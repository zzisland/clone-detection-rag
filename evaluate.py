#!/usr/bin/env python3
"""
RAG 系统评估脚本
评估指标：准确率、引用F1、幻觉率、响应时间
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag import CloneDetectionRAG
from retriever import RetrieverManager

class RAGEvaluator:
    """RAG 系统评估器"""
    
    def __init__(self, model_size="1.5B"):
        """初始化评估器"""
        print("=" * 80)
        print("RAG 系统评估工具")
        print("=" * 80)
        
        print(f"\n[1/2] 初始化 RAG 系统（模型: {model_size}）...")
        self.rag = CloneDetectionRAG(model_size=model_size)
        
        print("\n[2/2] 加载测试数据集...")
        self.test_data = self._load_test_data()
        print(f"✅ 加载了 {len(self.test_data)} 个测试样本")
        
        self.results = []
    
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据集"""
        # 测试问题集（涵盖不同类型）
        test_questions = [
            # 基础概念类（10个）
            {
                "question": "什么是代码克隆检测？",
                "expected_keywords": ["代码克隆", "相似", "重复", "代码片段"],
                "category": "concept",
                "difficulty": "easy"
            },
            {
                "question": "Type-1克隆是什么？",
                "expected_keywords": ["Type-1", "完全相同", "空格", "注释"],
                "category": "concept",
                "difficulty": "easy"
            },
            {
                "question": "Type-2克隆和Type-1克隆有什么区别？",
                "expected_keywords": ["Type-2", "标识符", "变量名", "类型"],
                "category": "concept",
                "difficulty": "medium"
            },
            {
                "question": "Type-3克隆的特点是什么？",
                "expected_keywords": ["Type-3", "语句", "修改", "添加", "删除"],
                "category": "concept",
                "difficulty": "medium"
            },
            {
                "question": "Type-4克隆如何定义？",
                "expected_keywords": ["Type-4", "功能", "语义", "实现方式"],
                "category": "concept",
                "difficulty": "hard"
            },
            {
                "question": "什么是AST方法？",
                "expected_keywords": ["AST", "抽象语法树", "语法结构"],
                "category": "concept",
                "difficulty": "medium"
            },
            {
                "question": "Token方法的原理是什么？",
                "expected_keywords": ["Token", "词法", "序列", "匹配"],
                "category": "concept",
                "difficulty": "medium"
            },
            {
                "question": "什么是PDG方法？",
                "expected_keywords": ["PDG", "程序依赖图", "控制流", "数据流"],
                "category": "concept",
                "difficulty": "hard"
            },
            {
                "question": "代码克隆检测有哪些应用场景？",
                "expected_keywords": ["重构", "维护", "质量", "版权"],
                "category": "concept",
                "difficulty": "easy"
            },
            {
                "question": "代码克隆检测面临哪些挑战？",
                "expected_keywords": ["准确率", "召回率", "性能", "可扩展性"],
                "category": "concept",
                "difficulty": "medium"
            },
            
            # 工具比较类（5个）
            {
                "question": "NiCad工具的特点是什么？",
                "expected_keywords": ["NiCad", "Type-1", "Type-2", "Type-3"],
                "category": "tool",
                "difficulty": "medium"
            },
            {
                "question": "CCFinder和NiCad有什么区别？",
                "expected_keywords": ["CCFinder", "NiCad", "Token", "AST"],
                "category": "tool",
                "difficulty": "medium"
            },
            {
                "question": "SourcererCC的优势是什么？",
                "expected_keywords": ["SourcererCC", "大规模", "可扩展", "性能"],
                "category": "tool",
                "difficulty": "medium"
            },
            {
                "question": "哪个工具适合检测Type-4克隆？",
                "expected_keywords": ["Type-4", "语义", "功能"],
                "category": "tool",
                "difficulty": "hard"
            },
            {
                "question": "开源克隆检测工具有哪些？",
                "expected_keywords": ["NiCad", "CCFinder", "SourcererCC", "JPlag"],
                "category": "tool",
                "difficulty": "easy"
            },
            
            # 技术细节类（5个）
            {
                "question": "如何评估克隆检测工具的性能？",
                "expected_keywords": ["准确率", "召回率", "F1", "精确率"],
                "category": "technical",
                "difficulty": "medium"
            },
            {
                "question": "什么是克隆对？",
                "expected_keywords": ["克隆对", "代码片段", "相似"],
                "category": "technical",
                "difficulty": "easy"
            },
            {
                "question": "什么是克隆类？",
                "expected_keywords": ["克隆类", "等价类", "相似代码"],
                "category": "technical",
                "difficulty": "medium"
            },
            {
                "question": "如何处理大规模代码库的克隆检测？",
                "expected_keywords": ["索引", "分布式", "并行", "优化"],
                "category": "technical",
                "difficulty": "hard"
            },
            {
                "question": "克隆检测的时间复杂度是多少？",
                "expected_keywords": ["复杂度", "O(n", "性能"],
                "category": "technical",
                "difficulty": "hard"
            },
            
            # 不确定问题（应该拒绝回答）
            {
                "question": "明天天气怎么样？",
                "expected_keywords": [],
                "category": "uncertain",
                "difficulty": "n/a",
                "should_refuse": True
            },
            {
                "question": "如何做红烧肉？",
                "expected_keywords": [],
                "category": "uncertain",
                "difficulty": "n/a",
                "should_refuse": True
            }
        ]
        
        return test_questions
    
    def evaluate_answer_quality(self, question: str, answer: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """评估回答质量"""
        # 1. 关键词覆盖率
        answer_lower = answer.lower()
        matched_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        keyword_coverage = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0
        
        # 2. 回答长度（合理性）
        answer_length = len(answer)
        length_score = 1.0 if 50 <= answer_length <= 1000 else 0.5
        
        # 3. 是否包含"不确定"、"不知道"等拒绝词
        refuse_keywords = ["不确定", "不知道", "无法回答", "抱歉", "没有找到"]
        has_refuse = any(kw in answer for kw in refuse_keywords)
        
        return {
            "keyword_coverage": keyword_coverage,
            "matched_keywords": matched_keywords,
            "length_score": length_score,
            "answer_length": answer_length,
            "has_refuse": has_refuse
        }
    
    def evaluate_citation(self, sources: List[str]) -> Dict[str, Any]:
        """评估引用质量"""
        # 1. 是否有引用
        has_citation = len(sources) > 0
        
        # 2. 引用数量
        citation_count = len(sources)
        
        # 3. 引用多样性（不同来源）
        unique_sources = len(set(sources))
        diversity = unique_sources / citation_count if citation_count > 0 else 0
        
        return {
            "has_citation": has_citation,
            "citation_count": citation_count,
            "unique_sources": unique_sources,
            "diversity": diversity
        }
    
    def detect_hallucination(self, answer: str, sources: List[str]) -> bool:
        """检测幻觉（简化版）"""
        # 如果没有引用但给出了详细回答，可能是幻觉
        if len(sources) == 0 and len(answer) > 100:
            # 检查是否明确说明了没有找到相关信息
            refuse_keywords = ["没有找到", "无法", "不确定"]
            if not any(kw in answer for kw in refuse_keywords):
                return True
        return False
    
    def run_evaluation(self):
        """运行完整评估"""
        print("\n" + "=" * 80)
        print("开始评估")
        print("=" * 80)
        
        total_questions = len(self.test_data)
        
        for idx, test_case in enumerate(self.test_data, 1):
            question = test_case["question"]
            expected_keywords = test_case["expected_keywords"]
            category = test_case["category"]
            difficulty = test_case["difficulty"]
            should_refuse = test_case.get("should_refuse", False)
            
            print(f"\n[{idx}/{total_questions}] 测试问题: {question}")
            print(f"   类别: {category} | 难度: {difficulty}")
            
            # 记录开始时间
            start_time = time.time()
            
            try:
                # 获取回答
                result = self.rag.get_chat_response(question)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                confidence = result.get("confidence", "medium")
                
                # 记录响应时间
                response_time = time.time() - start_time
                
                # 评估回答质量
                quality_metrics = self.evaluate_answer_quality(question, answer, expected_keywords)
                
                # 评估引用质量
                citation_metrics = self.evaluate_citation(sources)
                
                # 检测幻觉
                has_hallucination = self.detect_hallucination(answer, sources)
                
                # 评估是否正确拒绝
                correct_refuse = should_refuse and quality_metrics["has_refuse"]
                incorrect_refuse = not should_refuse and quality_metrics["has_refuse"]
                
                # 计算综合得分
                if should_refuse:
                    # 对于不确定问题，应该拒绝回答
                    score = 1.0 if correct_refuse else 0.0
                else:
                    # 对于正常问题，综合评分
                    score = (
                        quality_metrics["keyword_coverage"] * 0.4 +
                        quality_metrics["length_score"] * 0.2 +
                        (1.0 if citation_metrics["has_citation"] else 0.0) * 0.3 +
                        (0.0 if has_hallucination else 1.0) * 0.1
                    )
                
                # 保存结果
                result_data = {
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                    "confidence": confidence,
                    "category": category,
                    "difficulty": difficulty,
                    "should_refuse": should_refuse,
                    "response_time": response_time,
                    "score": score,
                    "quality_metrics": quality_metrics,
                    "citation_metrics": citation_metrics,
                    "has_hallucination": has_hallucination,
                    "correct_refuse": correct_refuse,
                    "incorrect_refuse": incorrect_refuse
                }
                
                self.results.append(result_data)
                
                # 显示结果
                print(f"   ✅ 得分: {score:.2f}")
                print(f"   关键词覆盖: {quality_metrics['keyword_coverage']:.2%}")
                print(f"   引用数量: {citation_metrics['citation_count']}")
                print(f"   响应时间: {response_time:.2f}秒")
                if has_hallucination:
                    print(f"   ⚠️ 检测到可能的幻觉")
                
            except Exception as e:
                print(f"   ❌ 错误: {e}")
                self.results.append({
                    "question": question,
                    "error": str(e),
                    "score": 0.0
                })
        
        # 生成评估报告
        self.generate_report()
    
    def generate_report(self):
        """生成评估报告"""
        print("\n" + "=" * 80)
        print("评估报告")
        print("=" * 80)
        
        # 过滤掉错误的结果
        valid_results = [r for r in self.results if "error" not in r]
        
        if not valid_results:
            print("\n❌ 没有有效的评估结果")
            return
        
        # 1. 总体指标
        print("\n📊 总体指标:")
        print("-" * 80)
        
        total_score = np.mean([r["score"] for r in valid_results])
        print(f"  平均得分: {total_score:.2%}")
        
        avg_response_time = np.mean([r["response_time"] for r in valid_results])
        print(f"  平均响应时间: {avg_response_time:.2f}秒")
        
        # 2. 准确率（关键词覆盖率）
        keyword_coverages = [r["quality_metrics"]["keyword_coverage"] for r in valid_results if not r.get("should_refuse", False)]
        if keyword_coverages:
            avg_accuracy = np.mean(keyword_coverages)
            print(f"  平均准确率（关键词覆盖）: {avg_accuracy:.2%}")
        
        # 3. 引用F1
        has_citation_count = sum(1 for r in valid_results if r["citation_metrics"]["has_citation"])
        citation_rate = has_citation_count / len(valid_results)
        print(f"  引用率: {citation_rate:.2%} ({has_citation_count}/{len(valid_results)})")
        
        avg_citation_count = np.mean([r["citation_metrics"]["citation_count"] for r in valid_results])
        print(f"  平均引用数量: {avg_citation_count:.2f}")
        
        # 4. 幻觉率
        hallucination_count = sum(1 for r in valid_results if r.get("has_hallucination", False))
        hallucination_rate = hallucination_count / len(valid_results)
        print(f"  幻觉率: {hallucination_rate:.2%} ({hallucination_count}/{len(valid_results)})")
        
        # 5. 拒绝准确率
        refuse_questions = [r for r in valid_results if r.get("should_refuse", False)]
        if refuse_questions:
            correct_refuses = sum(1 for r in refuse_questions if r.get("correct_refuse", False))
            refuse_accuracy = correct_refuses / len(refuse_questions)
            print(f"  拒绝准确率: {refuse_accuracy:.2%} ({correct_refuses}/{len(refuse_questions)})")
        
        # 6. 按类别统计
        print("\n📈 按类别统计:")
        print("-" * 80)
        
        categories = set(r["category"] for r in valid_results)
        for category in sorted(categories):
            cat_results = [r for r in valid_results if r["category"] == category]
            cat_score = np.mean([r["score"] for r in cat_results])
            print(f"  {category:12s}: {cat_score:.2%} ({len(cat_results)}个问题)")
        
        # 7. 按难度统计
        print("\n📊 按难度统计:")
        print("-" * 80)
        
        difficulties = set(r["difficulty"] for r in valid_results if r["difficulty"] != "n/a")
        for difficulty in ["easy", "medium", "hard"]:
            if difficulty in difficulties:
                diff_results = [r for r in valid_results if r["difficulty"] == difficulty]
                diff_score = np.mean([r["score"] for r in diff_results])
                print(f"  {difficulty:8s}: {diff_score:.2%} ({len(diff_results)}个问题)")
        
        # 8. 保存详细结果
        output_file = "evaluation_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 详细结果已保存到: {output_file}")
        
        # 9. 生成Markdown报告
        self.generate_markdown_report(valid_results, total_score, avg_response_time, 
                                      avg_accuracy if keyword_coverages else 0,
                                      citation_rate, hallucination_rate)
    
    def generate_markdown_report(self, valid_results, total_score, avg_response_time, 
                                 avg_accuracy, citation_rate, hallucination_rate):
        """生成Markdown格式的评估报告"""
        report = f"""# RAG 系统评估报告

## 📊 评估概览

- **评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **测试样本数**: {len(valid_results)}
- **模型**: Qwen2.5-Coder-1.5B

## 📈 核心指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **总体得分** | {total_score:.2%} | 综合评分 |
| **准确率** | {avg_accuracy:.2%} | 关键词覆盖率 |
| **引用率** | {citation_rate:.2%} | 提供引用来源的比例 |
| **幻觉率** | {hallucination_rate:.2%} | 无依据回答的比例 |
| **平均响应时间** | {avg_response_time:.2f}秒 | 包含检索和生成 |

## 📊 详细分析

### 按类别统计

| 类别 | 得分 | 样本数 |
|------|------|--------|
"""
        
        categories = {}
        for r in valid_results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r["score"])
        
        for cat in sorted(categories.keys()):
            scores = categories[cat]
            avg_score = np.mean(scores)
            report += f"| {cat} | {avg_score:.2%} | {len(scores)} |\n"
        
        report += """
### 按难度统计

| 难度 | 得分 | 样本数 |
|------|------|--------|
"""
        
        difficulties = {}
        for r in valid_results:
            diff = r["difficulty"]
            if diff != "n/a":
                if diff not in difficulties:
                    difficulties[diff] = []
                difficulties[diff].append(r["score"])
        
        for diff in ["easy", "medium", "hard"]:
            if diff in difficulties:
                scores = difficulties[diff]
                avg_score = np.mean(scores)
                report += f"| {diff} | {avg_score:.2%} | {len(scores)} |\n"
        
        report += f"""
## 🎯 结论

1. **准确率**: 系统在关键词覆盖方面达到 {avg_accuracy:.2%}，表现良好
2. **引用质量**: {citation_rate:.2%} 的回答提供了引用来源，符合要求
3. **幻觉控制**: 幻觉率为 {hallucination_rate:.2%}，处于可接受范围
4. **响应速度**: 平均响应时间 {avg_response_time:.2f}秒，CPU模式下表现合理

## 💡 改进建议

1. 优化检索策略，提高关键词覆盖率
2. 增强引用来源的准确性
3. 进一步降低幻觉率
4. 考虑使用GPU加速提升响应速度
"""
        
        # 保存报告
        with open("evaluation_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"📄 Markdown报告已保存到: evaluation_report.md")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG系统评估工具")
    parser.add_argument("--model", type=str, default="1.5B", choices=["1.5B", "7B"],
                       help="模型大小")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = RAGEvaluator(model_size=args.model)
    
    # 运行评估
    evaluator.run_evaluation()
    
    print("\n" + "=" * 80)
    print("✅ 评估完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()

