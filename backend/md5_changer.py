"""
MD5修改模块 — 三种方式修改视频/图片的MD5值
1. 重编码：重新编码改变文件内容
2. 元数据修改：修改文件元数据
3. 尾部追加：文件末尾追加随机字节
"""

import os
import struct
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
import random
import string


def get_md5(filepath: str) -> str:
    """计算文件的MD5值"""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def get_file_info(filepath: str) -> dict:
    """获取文件基本信息"""
    path = Path(filepath)
    stat = path.stat()
    return {
        'path': str(path),
        'name': path.name,
        'suffix': path.suffix.lower(),
        'size': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'md5': get_md5(filepath),
        'is_video': path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'),
        'is_image': path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'),
    }


def method_append_bytes(filepath: str, output_dir: str = None) -> dict:
    """
    方式一：尾部追加随机字节
    优点：最快，不影响播放/显示
    """
    info = get_file_info(filepath)
    path = Path(filepath)
    
    # 生成输出文件名
    output_dir = Path(output_dir) if output_dir else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{path.stem}_md5_append{path.suffix}"
    output_path = output_dir / output_name
    
    # 读取原文件 + 追加字节
    with open(filepath, 'rb') as src, open(output_path, 'wb') as dst:
        dst.write(src.read())
        # 追加16字节随机数据和标记
        random_bytes = bytes([random.randint(0, 255) for _ in range(12)])
        marker = b'MD5MOD'
        dst.write(random_bytes + marker)
    
    new_info = get_file_info(str(output_path))
    return {
        'method': 'append',
        'original': info,
        'result': new_info,
        'output': str(output_path),
        'original_md5': info['md5'],
        'new_md5': new_info['md5'],
    }


def method_metadata(filepath: str, output_dir: str = None) -> dict:
    """
    方式二：修改元数据
    对JPEG/PNG修改EXIF，对视频修改creation_time等
    """
    from PIL import Image
    from mutagen.mp4 import MP4
    from mutagen import File as MutagenFile
    
    info = get_file_info(filepath)
    path = Path(filepath)
    
    output_dir = Path(output_dir) if output_dir else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{path.stem}_md5_meta{path.suffix}"
    output_path = output_dir / output_name
    
    # 先复制原文件
    import shutil
    shutil.copy2(filepath, output_path)
    
    if info['is_image']:
        try:
            img = Image.open(output_path)
            # 添加/修改EXIF评论
            exif = img.getexif()
            if exif is None:
                exif = img.Info.get('exif', b'')
            # 写入时间戳评论
            import piexif
            exif_dict = piexif.load(img.info.get('exif', b''))
            exif_dict['0th'][piexif.ImageIFD.ImageDescription] = f'MD5_MOD_{datetime.now().isoformat()}'
            exif_dict['Exif'][piexif.ExifIFD.UserComment] = f'Modified_{random.randint(100000,999999)}'.encode()
            exif_bytes = piexif.dump(exif_dict)
            img.save(output_path, exif=exif_bytes)
        except ImportError:
            # 没有piexif时的降级方案：简单修改像素
            img = img.convert('RGB')
            pixels = img.load()
            if img.width > 0 and img.height > 0:
                # 修改第一个像素（肉眼不可见）
                r, g, b = pixels[0, 0]
                pixels[0, 0] = (r, g, (b + 1) % 256 if b < 255 else b - 1)
            img.save(output_path)
    
    elif info['is_video']:
        try:
            # 使用mutagen修改视频元数据
            video = MutagenFile(output_path)
            if video is not None and hasattr(video, 'tags'):
                if video.tags is None:
                    video.add_tags()
                video['©cmt'] = f'MD5_MOD_{datetime.now().isoformat()}'
                video['©too'] = f'tool_{random.randint(100000,999999)}'
                video.save()
        except Exception:
            pass
    
    new_info = get_file_info(str(output_path))
    return {
        'method': 'metadata',
        'original': info,
        'result': new_info,
        'output': str(output_path),
        'original_md5': info['md5'],
        'new_md5': new_info['md5'],
    }


def method_reencode(filepath: str, output_dir: str = None, 
                    quality: str = 'medium', progress_callback=None) -> dict:
    """
    方式三：重新编码
    优点：彻底改变MD5，可调画质
    对视频用ffmpeg重新编码，对图片用Pillow重新保存
    """
    info = get_file_info(filepath)
    path = Path(filepath)
    
    output_dir = Path(output_dir) if output_dir else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{path.stem}_md5_reencode{path.suffix}"
    output_path = output_dir / output_name
    
    if info['is_video']:
        # ffmpeg重新编码视频
        quality_map = {
            'high':   24,   # CRF越低画质越好
            'medium': 28,
            'low':    35,
        }
        crf = quality_map.get(quality, 28)
        
        cmd = [
            'ffmpeg', '-y',
            '-i', filepath,
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-map_metadata', '0',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg重编码失败: {result.stderr[:500]}")
    
    elif info['is_image']:
        # 用Pillow重新保存（改变压缩参数）
        from PIL import Image
        img = Image.open(filepath)
        
        if path.suffix.lower() in ('.jpg', '.jpeg'):
            img.save(output_path, 'JPEG', quality=random.randint(85, 95))
        elif path.suffix.lower() == '.png':
            img.save(output_path, 'PNG', compress_level=random.randint(3, 7))
        elif path.suffix.lower() == '.webp':
            img.save(output_path, 'WEBP', quality=random.randint(80, 95))
        else:
            img.save(output_path)
    
    else:
        # 非视频非图片：用尾部追加
        return method_append_bytes(filepath, output_dir)
    
    new_info = get_file_info(str(output_path))
    return {
        'method': 'reencode',
        'original': info,
        'result': new_info,
        'output': str(output_path),
        'original_md5': info['md5'],
        'new_md5': new_info['md5'],
    }


def process_md5(filepath: str, methods: list = None, 
                output_dir: str = None, quality: str = 'medium') -> list:
    """
    统一入口：执行MD5修改
    
    Args:
        filepath: 输入文件路径
        methods: 方法列表 ['append', 'metadata', 'reencode']
        output_dir: 输出目录
        quality: 重编码画质
    
    Returns:
        处理结果列表
    """
    if methods is None:
        methods = ['append']
    
    results = []
    for method in methods:
        try:
            if method == 'append':
                r = method_append_bytes(filepath, output_dir)
            elif method == 'metadata':
                r = method_metadata(filepath, output_dir)
            elif method == 'reencode':
                r = method_reencode(filepath, output_dir, quality)
            else:
                continue
            r['success'] = True
            results.append(r)
        except Exception as e:
            results.append({
                'method': method,
                'success': False,
                'error': str(e),
                'original': get_file_info(filepath),
            })
    
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        fp = sys.argv[1]
        print(f"原始MD5: {get_md5(fp)}")
        results = process_md5(fp, methods=['append', 'metadata', 'reencode'],
                             output_dir='D:/fanben/媒体处理体系/输出/')
        for r in results:
            if r['success']:
                print(f"[{r['method']}] 旧MD5: {r['original_md5']} → 新MD5: {r['new_md5']}")
            else:
                print(f"[{r['method']}] 失败: {r.get('error')}")
