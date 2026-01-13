#!/usr/bin/env python3
"""
快速数据摄取脚本
优化了数据摄取速度，支持增量更新和批量处理
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.ingest import DataIngestor
from src.config import Config

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='快速数据摄取工具')
    parser.add_argument('--force', action='store_true', 
                      help='强制重新摄取所有数据')
    parser.add_argument('--batch-size', type=int, default=100,
                      help='批量处理大小 (默认: 100)')
    parser.add_argument('--chunk-size', type=int, default=2000,
                      help='文档分块大小 (默认: 2000)')
    parser.add_argument('--chunk-overlap', type=int, default=100,
                      help='分块重叠大小 (默认: 100)')
    
    args = parser.parse_args()
    
    # 更新配置
    Config.CHUNK_SIZE = args.chunk_size
    Config.CHUNK_OVERLAP = args.chunk_overlap
    
    print("=== 快速数据摄取工具 ===")
    print(f"分块大小: {Config.CHUNK_SIZE}")
    print(f"分块重叠: {Config.CHUNK_OVERLAP}")
    print(f"批量大小: {args.batch_size}")
    print(f"强制重新摄取: {args.force}")
    print()
    
    # 创建数据摄取器
    ingestor = DataIngestor()
    
    try:
        # 执行数据摄取
        vector_store = ingestor.ingest_all_data(force_refresh=args.force)
        
        if vector_store:
            print("\n=== 摄取完成 ===")
            print("✅ 数据摄取成功完成")
            print(f"📁 数据库位置: {Config.CHROMA_PERSIST_DIRECTORY}")
            
            # 显示统计信息
            try:
                collection = vector_store._collection
                count = collection.count()
                print(f"📊 向量数量: {count}")
            except:
                print("📊 无法获取向量数量")
        else:
            print("❌ 数据摄取失败")
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断了数据摄取")
    except Exception as e:
        print(f"\n❌ 数据摄取出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
