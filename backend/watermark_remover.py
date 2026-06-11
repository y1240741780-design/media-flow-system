"""
去水印模块 — 支持静态和动态水印去除
使用 OpenCV 进行水印检测、追踪和修复
"""

import os
import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import shutil


def detect_watermark_region(frame: np.ndarray, roi: tuple = None) -> tuple:
    """
    检测水印区域
    - roi: 手动指定的区域 (x, y, w, h)
    - 自动检测：基于边缘密度和颜色一致性
    """
    if roi is not None:
        return roi  # 使用手动指定的区域
    
    # 自动检测（简单策略：检测角落区域的固定纹理）
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检查四个角落的边缘密度
    corners = {
        'top_left':     gray[0:h//4, 0:w//4],
        'top_right':    gray[0:h//4, 3*w//4:w],
        'bottom_left':  gray[3*h//4:h, 0:w//4],
        'bottom_right': gray[3*h//4:h, 3*w//4:w],
    }
    
    best_corner = None
    best_density = -1
    
    for name, region in corners.items():
        # 计算边缘密度
        edges = cv2.Canny(region, 50, 150)
        density = np.sum(edges > 0) / (region.shape[0] * region.shape[1])
        if density > best_density and density > 0.05:
            best_density = density
            best_corner = name
    
    # 返回检测到的角落区域
    if best_corner == 'top_left':
        return (0, 0, w//4, h//4)
    elif best_corner == 'top_right':
        return (3*w//4, 0, w//4, h//4)
    elif best_corner == 'bottom_left':
        return (0, 3*h//4, w//4, h//4)
    elif best_corner == 'bottom_right':
        return (3*w//4, 3*h//4, w//4, h//4)
    
    return None


def track_watermark(video_path: str, roi: tuple) -> list:
    """
    在视频中追踪动态水印的位置
    使用模板匹配在第一帧的水印区域和后续帧之间进行追踪
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, first_frame = cap.read()
    if not ret:
        return []
    
    x, y, w, h = roi
    template = first_frame[y:y+h, x:x+w]
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    positions = [(x, y, w, h)]  # 第一帧位置
    
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 模板匹配找水印位置
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.5:
            positions.append((max_loc[0], max_loc[1], w, h))
        else:
            positions.append((x, y, w, h))  # 找不到就用上一帧位置
        
        frame_idx += 1
    
    cap.release()
    return positions


def remove_watermark_frame(frame: np.ndarray, roi: tuple, method: str = 'inpaint') -> np.ndarray:
    """
    从单帧中去除水印
    
    Args:
        frame: 视频帧
        roi: 水印区域 (x, y, w, h)
        method: 'inpaint' | 'blur' | 'median_stack'
    """
    x, y, w, h = roi
    x, y = max(0, x), max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    
    if method == 'inpaint':
        # 创建水印区域的mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # 略微扩展mask区域
        margin = 5
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        mask[y1:y2, x1:x2] = 255
        
        # 使用Navier-Stokes inpainting
        result = cv2.inpaint(frame, mask, 5, cv2.INPAINT_NS)
        return result
    
    elif method == 'blur':
        # 对水印区域高斯模糊
        result = frame.copy()
        roi_area = result[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(roi_area, (31, 31), 0)
        result[y:y+h, x:x+w] = blurred
        return result
    
    else:
        return frame


def remove_watermark_static(video_path: str, output_path: str,
                           roi: tuple, method: str = 'inpaint',
                           progress_callback=None) -> dict:
    """
    去除静态水印（位置不动的水印）
    直接对每帧的水印区域做inpainting
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 用ffmpeg重新编码（opencv写视频质量不够好）
    # 先用opencv处理 + 临时文件，再用ffmpeg合并音轨
    
    temp_video = output_path + '.temp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = remove_watermark_frame(frame, roi, method)
        out.write(processed)
        frame_idx += 1
        
        if progress_callback and frame_idx % 30 == 0:
            progress_callback(frame_idx / total_frames * 100)
    
    cap.release()
    out.release()
    
    # 用ffmpeg合并原始音频 + 处理后视频
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_video,
        '-i', video_path,
        '-c:v', 'libx264', '-crf', '23',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-shortest',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, timeout=600)
    
    # 清理临时文件
    if os.path.exists(temp_video):
        os.remove(temp_video)
    
    return {
        'method': f'static_{method}',
        'output': output_path,
        'total_frames': total_frames,
        'processed_frames': frame_idx,
    }


def remove_watermark_dynamic(video_path: str, output_path: str,
                            roi: tuple, method: str = 'inpaint',
                            progress_callback=None) -> dict:
    """
    去除动态水印（移动水印）
    先追踪水印位置，再逐帧去除
    """
    # 追踪水印
    positions = track_watermark(video_path, roi)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    temp_video = output_path + '.temp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(positions):
            pos = positions[frame_idx]
            processed = remove_watermark_frame(frame, pos, method)
        else:
            processed = frame
        
        out.write(processed)
        frame_idx += 1
        
        if progress_callback and frame_idx % 30 == 0:
            progress_callback(frame_idx / total_frames * 100)
    
    cap.release()
    out.release()
    
    # 合并音频
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_video,
        '-i', video_path,
        '-c:v', 'libx264', '-crf', '23',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-shortest',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, timeout=600)
    
    if os.path.exists(temp_video):
        os.remove(temp_video)
    
    return {
        'method': f'dynamic_{method}',
        'output': output_path,
        'total_frames': total_frames,
        'processed_frames': frame_idx,
        'watermark_positions': len(positions),
    }


def remove_watermark_image(image_path: str, output_path: str,
                          roi: tuple, method: str = 'inpaint') -> dict:
    """去除图片水印"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    result = remove_watermark_frame(img, roi, method)
    cv2.imwrite(output_path, result)
    
    return {
        'method': f'image_{method}',
        'output': output_path,
    }


def process_watermark(filepath: str, output_dir: str = None,
                     roi: tuple = None, watermark_type: str = 'static',
                     method: str = 'inpaint', progress_callback=None) -> dict:
    """
    统一入口：去水印
    
    Args:
        filepath: 输入文件
        output_dir: 输出目录
        roi: 水印区域 (x, y, w, h)，None则自动检测
        watermark_type: 'static' | 'dynamic'
        method: 'inpaint' | 'blur'
    """
    path = Path(filepath)
    output_dir = Path(output_dir) if output_dir else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{path.stem}_nowatermark{path.suffix}"
    output_path = str(output_dir / output_name)
    
    is_video = path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv')
    is_image = path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')
    
    # 自动检测水印位置
    if roi is None and is_video:
        cap = cv2.VideoCapture(filepath)
        ret, first_frame = cap.read()
        cap.release()
        if ret:
            roi = detect_watermark_region(first_frame)
    
    if is_video and watermark_type == 'static':
        result = remove_watermark_static(filepath, output_path, roi, method, 
                                         progress_callback)
    elif is_video and watermark_type == 'dynamic':
        result = remove_watermark_dynamic(filepath, output_path, roi, method,
                                          progress_callback)
    elif is_image:
        result = remove_watermark_image(filepath, output_path, roi, method)
    else:
        result = {'error': '不支持的文件类型'}
    
    result['success'] = 'error' not in result
    result['output'] = output_path
    result['original'] = str(filepath)
    result['roi'] = roi
    
    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        fp = sys.argv[1]
        print(f"处理文件: {fp}")
        result = process_watermark(fp, output_dir='D:/fanben/媒体处理体系/输出/',
                                   watermark_type='static')
        print(result)
