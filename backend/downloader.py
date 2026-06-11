"""
在线下载模块 — 从各平台下载视频/图片
基于 yt-dlp，支持抖音/快手/B站/YouTube 等主流平台
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime


PLATFORM_CONFIG = {
    'douyin': {
        'name': '抖音',
        'domains': ['douyin.com', 'iesdouyin.com'],
    },
    'kuaishou': {
        'name': '快手',
        'domains': ['kuaishou.com', 'gifshow.com'],
    },
    'bilibili': {
        'name': 'B站',
        'domains': ['bilibili.com'],
    },
    'youtube': {
        'name': 'YouTube',
        'domains': ['youtube.com', 'youtu.be'],
    },
    'xiaohongshu': {
        'name': '小红书',
        'domains': ['xiaohongshu.com', 'xhslink.com'],
    },
    'weibo': {
        'name': '微博',
        'domains': ['weibo.com'],
    },
}


def detect_platform(url: str) -> str:
    """根据URL自动识别平台"""
    url_lower = url.lower()
    for key, config in PLATFORM_CONFIG.items():
        for domain in config['domains']:
            if domain in url_lower:
                return key
    return 'unknown'


def download_media(url: str, output_dir: str = None,
                   platform: str = None, options: dict = None) -> dict:
    """
    下载视频/图片
    
    Args:
        url: 视频/图片链接
        output_dir: 输出目录
        platform: 平台标识（自动检测）
        options: 额外选项
            - format: 'best' | 'worst' | 'mp4' | 'audio'
            - max_size_mb: 最大文件大小
    """
    options = options or {}
    output_dir = Path(output_dir) if output_dir else Path('D:/fanben/媒体处理体系/输出/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if platform is None:
        platform = detect_platform(url)
    
    # 输出模板
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_template = str(output_dir / f'%(title)s_{timestamp}.%(ext)s')
    
    # 构建yt-dlp命令
    cmd = [
        'yt-dlp',
        '--no-playlist',
        '--no-warnings',
        '--print-json',
        '-o', output_template,
    ]
    
    # 格式选择
    fmt = options.get('format', 'best')
    if fmt == 'best':
        cmd.extend(['-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'])
    elif fmt == 'mp4':
        cmd.extend(['-f', 'best[ext=mp4]'])
    elif fmt == 'audio':
        cmd.extend(['-f', 'bestaudio', '-x', '--audio-format', 'mp3'])
    
    # 大小限制
    if options.get('max_size_mb'):
        cmd.extend(['--max-filesize', f'{options["max_size_mb"]}M'])
    
    # 平台特定参数
    if platform == 'douyin':
        cmd.extend(['--cookies-from-browser', 'chrome'])  # 需要浏览器cookie
    
    cmd.append(url)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': result.stderr[:500] if result.stderr else '下载失败',
                'platform': platform,
            }
        
        # 解析yt-dlp输出的JSON
        output_lines = result.stdout.strip().split('\n')
        file_info = None
        for line in output_lines:
            try:
                data = json.loads(line)
                file_info = {
                    'title': data.get('title', ''),
                    'duration': data.get('duration', 0),
                    'resolution': f"{data.get('width', 0)}x{data.get('height', 0)}",
                    'filesize_mb': round(data.get('filesize', 0) / (1024 * 1024), 2) if data.get('filesize') else 0,
                    'ext': data.get('ext', ''),
                    'uploader': data.get('uploader', ''),
                    'description': (data.get('description', '') or '')[:200],
                    'webpage_url': data.get('webpage_url', url),
                }
                break
            except json.JSONDecodeError:
                continue
        
        # 查找实际下载的文件
        downloaded_files = sorted(
            output_dir.glob(f'*{timestamp}*'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        downloaded_path = str(downloaded_files[0]) if downloaded_files else ''
        
        return {
            'success': True,
            'platform': PLATFORM_CONFIG.get(platform, {}).get('name', platform),
            'file_info': file_info,
            'downloaded_path': downloaded_path,
            'output_dir': str(output_dir),
        }
    
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': '下载超时（超过120秒）',
            'platform': platform,
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'platform': platform,
        }


def batch_download(urls: list, output_dir: str = None, options: dict = None) -> list:
    """批量下载"""
    results = []
    for url in urls:
        result = download_media(url.strip(), output_dir, options=options)
        results.append(result)
    return results


def get_platform_info(platform: str = None) -> dict:
    """获取支持的平台列表"""
    if platform and platform in PLATFORM_CONFIG:
        return PLATFORM_CONFIG[platform]
    return PLATFORM_CONFIG


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        url = sys.argv[1]
        result = download_media(url)
        print(json.dumps(result, ensure_ascii=False, indent=2))
