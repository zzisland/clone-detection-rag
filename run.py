#!/usr/bin/env python3
"""
代码克隆检测RAG助手启动脚本
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """检查环境要求"""
    print("🔍 检查环境要求...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    # 检查必要的包
    required_packages = [
        'streamlit', 'langchain', 'chromadb', 
        'openai', 'dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少必要的包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 环境要求检查通过")
    return True

def check_config():
    """检查配置文件"""
    print("🔧 检查配置文件...")
    
    # 检查.env文件
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️ 未找到.env文件")
        if Path('.env.example').exists():
            print("正在创建.env文件...")
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ 已创建.env文件，请编辑并添加你的API Key")
            return False
        else:
            print("❌ 未找到.env.example文件")
            return False
    
    # 检查API Key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your_openai_api_key_here':
        print("❌ 请在.env文件中设置有效的OPENAI_API_KEY")
        return False
    
    print("✅ 配置文件检查通过")
    return True

def check_data():
    """检查数据目录"""
    print("📚 检查数据目录...")
    
    data_dirs = [
        'data/papers',
        'data/tools_docs', 
        'data/project_docs',
        'data/examples'
    ]
    
    missing_dirs = []
    for dir_path in data_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ 缺少数据目录: {', '.join(missing_dirs)}")
        return False
    
    # 检查是否有文档
    doc_count = 0
    for dir_path in data_dirs:
        for file_path in Path(dir_path).rglob('*'):
            if file_path.is_file():
                doc_count += 1
    
    if doc_count == 0:
        print("⚠️ 数据目录为空，请添加文档文件")
        return False
    
    print(f"✅ 找到 {doc_count} 个文档文件")
    return True

def ingest_data():
    """摄取数据"""
    print("📥 开始摄取数据...")
    
    try:
        sys.path.append('src')
        from ingest import DataIngestor
        
        ingestor = DataIngestor()
        vector_store = ingestor.ingest_all_data()
        
        if vector_store:
            print("✅ 数据摄取完成")
            return True
        else:
            print("❌ 数据摄取失败")
            return False
            
    except Exception as e:
        print(f"❌ 数据摄取出错: {e}")
        return False

def main():
    """主函数"""
    print("🚀 启动代码克隆检测RAG助手")
    print("=" * 50)
    
    # 检查环境
    if not check_requirements():
        return
    
    # 检查配置
    if not check_config():
        return
    
    # 检查数据
    if not check_data():
        return
    
    # 询问是否重新摄取数据
    import streamlit as st
    from streamlit.runtime.scriptrunner import RerunData, RerunException
    
    try:
        # 尝试导入streamlit来检查是否已经摄取数据
        sys.path.append('src')
        from retriever import RetrieverManager
        
        manager = RetrieverManager()
        if not manager.load_vector_store():
            print("⚠️ 向量数据库不存在，需要先摄取数据")
            if not ingest_data():
                return
        else:
            print("✅ 向量数据库已存在")
            
            # 询问是否重新摄取
            response = input("是否重新摄取数据？(y/N): ").lower()
            if response == 'y':
                if not ingest_data():
                    return
    
    except Exception as e:
        print(f"⚠️ 检查向量数据库时出错: {e}")
        print("将进行数据摄取...")
        if not ingest_data():
            return
    
    # 启动Streamlit应用
    print("🌐 启动Web界面...")
    print("=" * 50)
    print("应用将在浏览器中打开: http://localhost:8501")
    print("按 Ctrl+C 停止应用")
    print("=" * 50)
    
    try:
        import subprocess
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动应用失败: {e}")

if __name__ == "__main__":
    main()
