"""
媒体处理体系 — 启动脚本
用法: python start.py
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent

def main():
    print("=" * 50)
    print("  媒体处理体系 — Stitch Flow")
    print("  启动中...")
    print("=" * 50)
    
    server_path = BASE_DIR / 'backend' / 'server.py'
    
    if not server_path.exists():
        print(f"[错误] 找不到服务器文件: {server_path}")
        sys.exit(1)
    
    print(f"  后端: {server_path}")
    print(f"  访问: http://localhost:5000")
    print("=" * 50)
    
    # 启动Flask服务器
    os.chdir(str(BASE_DIR / 'backend'))
    subprocess.run([sys.executable, str(server_path)])

if __name__ == '__main__':
    main()
