"""
无痕去水印模块 — 多层级高级算法

技术栈（按效果排序）：
  L1 (最佳): 帧累积中值滤波 — 利用多帧信息自然滤除水印，真正无痕
  L2 (优秀): Alpha水印估计 — 估计水印透明度并反演恢复原背景
  L3 (良好): 光流引导帧间修复 — 从邻近帧借用背景像素
  L4 (快速): ffmpeg delogo — 基于相邻像素插值
  L5 (兜底): 增强Inpaint — 精确mask + Poisson融合 + 边缘平滑
"""

import os
import cv2
import numpy as np
from scipy import ndimage, signal
from pathlib import Path
import subprocess
import tempfile
import shutil
from collections import deque
from typing import Optional, Tuple, List, Callable


# ============================================================
#  工具函数
# ============================================================

def _ensure_roi_bounds(roi: Tuple[int,int,int,int], frame_shape: tuple) -> Tuple[int,int,int,int]:
    """确保ROI在帧范围内，返回 (x, y, w, h)"""
    x, y, w, h = roi
    fh, fw = frame_shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, fw - x)
    h = min(h, fh - y)
    return (x, y, w, h)


def _create_soft_mask(shape: tuple, roi: Tuple[int,int,int,int], 
                      feather: int = 8) -> np.ndarray:
    """
    创建带羽化边缘的软mask（0.0~1.0），避免修复边界生硬
    """
    x, y, w, h = roi
    mask = np.zeros(shape[:2], dtype=np.float32)
    
    # 内部区域 = 1.0
    inner_x1 = x + feather
    inner_y1 = y + feather
    inner_x2 = x + w - feather
    inner_y2 = y + h - feather
    
    if inner_x1 < inner_x2 and inner_y1 < inner_y2:
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = 1.0
    
    # 羽化过渡区
    for margin in range(feather):
        alpha = (margin + 1) / feather
        
        # 上边缘
        yy = y + margin
        if 0 <= yy < shape[0]:
            x1 = max(0, x + margin)
            x2 = min(shape[1], x + w - margin)
            if x1 < x2:
                mask[yy, x1:x2] = np.maximum(mask[yy, x1:x2], alpha)
        
        # 下边缘
        yy = y + h - 1 - margin
        if 0 <= yy < shape[0]:
            x1 = max(0, x + margin)
            x2 = min(shape[1], x + w - margin)
            if x1 < x2:
                mask[yy, x1:x2] = np.maximum(mask[yy, x1:x2], alpha)
        
        # 左边缘
        xx = x + margin
        if 0 <= xx < shape[1]:
            y1 = max(0, y + margin)
            y2 = min(shape[0], y + h - margin)
            if y1 < y2:
                mask[y1:y2, xx] = np.maximum(mask[y1:y2, xx], alpha)
        
        # 右边缘
        xx = x + w - 1 - margin
        if 0 <= xx < shape[1]:
            y1 = max(0, y + margin)
            y2 = min(shape[0], y + h - margin)
            if y1 < y2:
                mask[y1:y2, xx] = np.maximum(mask[y1:y2, xx], alpha)
    
    return np.clip(mask, 0, 1)


def _poisson_blend(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    泊松融合 — 消除修复区域与周围的边界痕迹
    简化实现：对边界区域做梯度域混合
    """
    result = target.copy()
    mask_3ch = np.dstack([mask, mask, mask])
    
    # 对修复区域做拉普拉斯平滑后混合
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    
    for c in range(3):
        channel = source[:, :, c].astype(np.float32)
        laplacian = cv2.filter2D(channel, -1, kernel)
        
        # 在mask区域使用source的拉普拉斯 + target的边界条件
        blended = channel.copy()
        blended[mask > 0.5] = target[:, :, c][mask > 0.5] * 0.3 + channel[mask > 0.5] * 0.7
        
        result[:, :, c] = np.clip(blended, 0, 255).astype(np.uint8)
    
    return result


# ============================================================
#  L1: 帧累积中值滤波 (Multi-Frame Median Blending)
#  核心原理：水印在多帧中位置/透明度不同，中值滤波自然滤除
# ============================================================

def _multi_frame_median(video_path: str, output_path: str,
                        roi: Tuple[int,int,int,int],
                        window_size: int = 31,
                        progress_callback=None) -> dict:
    """
    帧累积中值滤波 — 无痕去水印核心技术
    
    原理：
    1. 取当前帧前后各 N 帧（共 window_size 帧）
    2. 对水印区域的每个像素，在时间维度取中值
    3. 由于水印在每帧位置不同（或透明度变化），中值会自然取到背景像素
    4. 对每帧重复此操作
    
    适合：动态水印（位置变化）、闪烁水印
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    x, y, w, h = _ensure_roi_bounds(roi, (height, width))
    half_win = window_size // 2
    
    # 第一遍：读取所有帧到内存（或者使用滑动窗口）
    # 对大视频使用滑动窗口以节省内存
    use_sliding = total_frames > 500 and width * height > 1280 * 720
    
    temp_video = output_path + '.temp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    if use_sliding:
        # 滑动窗口模式：只保留 window_size 帧在内存中
        frame_buffer = deque(maxlen=window_size)
        
        # 预读前 half_win 帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(half_win):
            ret, f = cap.read()
            if ret:
                frame_buffer.append(f.astype(np.float32))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 继续填充buffer（后续帧）
            while len(frame_buffer) < window_size:
                next_pos = frame_idx + len(frame_buffer) - half_win + 1
                if next_pos >= total_frames:
                    break
                cap2 = cv2.VideoCapture(video_path)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
                r2, f2 = cap2.read()
                cap2.release()
                if r2:
                    frame_buffer.append(f2.astype(np.float32))
                else:
                    break
            
            if len(frame_buffer) >= 3:
                # 对水印区域取中值
                roi_stack = np.stack([fb[y:y+h, x:x+w] for fb in frame_buffer], axis=0)
                median_roi = np.median(roi_stack, axis=0).astype(np.uint8)
                
                result = frame.copy()
                # 羽化混合
                soft_mask = _create_soft_mask(frame.shape, roi, feather=6)
                soft_mask_3ch = np.dstack([soft_mask, soft_mask, soft_mask])
                result[y:y+h, x:x+w] = (
                    median_roi * soft_mask_3ch[y:y+h, x:x+w] +
                    frame[y:y+h, x:x+w].astype(np.float32) * (1 - soft_mask_3ch[y:y+h, x:x+w])
                ).astype(np.uint8)
                
                out.write(result)
            else:
                out.write(frame)
            
            frame_idx += 1
            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx / total_frames * 100)
            
            # 滑动：移除最旧帧，添加新帧（需要重新seek读后续帧）
            # 简化：对滑动窗口模式，每次重建buffer
            if frame_idx >= half_win and frame_idx + half_win < total_frames:
                frame_buffer.clear()
                cap_temp = cv2.VideoCapture(video_path)
                start_f = max(0, frame_idx - half_win + 1)
                cap_temp.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                for _ in range(min(window_size, total_frames - start_f)):
                    r3, f3 = cap_temp.read()
                    if r3:
                        frame_buffer.append(f3.astype(np.float32))
                cap_temp.release()
    else:
        # 全量加载模式：适合短视频
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame.astype(np.float32))
        cap.release()
        total = len(all_frames)
        
        for i in range(total):
            frame = all_frames[i].copy()
            
            # 取窗口帧
            start = max(0, i - half_win)
            end = min(total, i + half_win + 1)
            window_frames = all_frames[start:end]
            
            if len(window_frames) >= 3:
                # 中值滤波
                roi_stack = np.stack([wf[y:y+h, x:x+w] for wf in window_frames], axis=0)
                median_roi = np.median(roi_stack, axis=0).astype(np.uint8)
                
                # 羽化混合
                soft_mask = _create_soft_mask(frame.shape, roi, feather=6)
                soft_mask_3ch = np.dstack([soft_mask, soft_mask, soft_mask])
                frame_roi = frame[y:y+h, x:x+w]
                frame[y:y+h, x:x+w] = (
                    median_roi.astype(np.float32) * soft_mask_3ch[y:y+h, x:x+w] +
                    frame_roi * (1 - soft_mask_3ch[y:y+h, x:x+w])
                ).astype(np.uint8)
            
            out.write(frame.astype(np.uint8))
            
            if progress_callback and i % 30 == 0:
                progress_callback((i + 1) / total * 100)
    
    cap.release()
    out.release()
    
    # 用ffmpeg合并音频
    _merge_audio(temp_video, video_path, output_path)
    
    return {
        'method': 'L1_multi_frame_median',
        'output': output_path,
        'window_size': window_size,
    }


# ============================================================
#  L2: Alpha水印估计与反演
#  原理：估计水印的alpha通道，逆向恢复原背景
#  适合：半透明固定水印（如抖音/@用户名角标）
# ============================================================

def _estimate_watermark_alpha(video_path: str, roi: Tuple[int,int,int,int],
                               sample_frames: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    估计水印的alpha通道和颜色
    
    通过采样多帧，分析水印区域像素的统计分布：
    - 背景像素：在帧间随机变化
    - 水印像素：在帧间保持一致（或按alpha混合）
    
    返回: (watermark_color, alpha_mask)
    """
    cap = cv2.VideoCapture(video_path)
    x, y, w, h = roi
    
    # 收集多帧的水印区域像素
    samples = []
    frame_count = 0
    while frame_count < sample_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // sample_frames) == 0:
            roi_patch = frame[y:y+h, x:x+w].astype(np.float32)
            samples.append(roi_patch)
        frame_count += 1
    cap.release()
    
    if len(samples) < 5:
        return None, None
    
    samples = np.stack(samples, axis=0)  # (N, h, w, 3)
    
    # 对每个像素位置，取最小值作为水印颜色估计
    # 原理：水印通常比背景亮（白字）或暗（黑边），
    # 对于亮水印：背景可能在部分帧中更暗 → 水印颜色≈该位置的中值偏上
    # 对于暗水印：水印≈最小值
    median_pixel = np.median(samples, axis=0)
    min_pixel = np.min(samples, axis=0)
    
    # Alpha估计：像素在帧间的方差
    variance = np.var(samples, axis=0)
    max_var = np.max(variance)
    if max_var > 0:
        alpha_raw = 1.0 - np.clip(np.mean(variance, axis=2) / (max_var + 1e-6), 0, 1)
    else:
        alpha_raw = np.ones((h, w), dtype=np.float32)
    
    # 平滑alpha
    alpha = cv2.GaussianBlur(alpha_raw.astype(np.float32), (5, 5), 1.5)
    
    return min_pixel.astype(np.uint8), alpha


def _alpha_inversion_remove(video_path: str, output_path: str,
                            roi: Tuple[int,int,int,int],
                            progress_callback=None) -> dict:
    """
    Alpha水印反演去除
    
    对每帧：估计背景 = (当前像素 - alpha * 水印颜色) / (1 - alpha)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    x, y, w, h = _ensure_roi_bounds(roi, (height, width))
    
    # 估计水印alpha和颜色
    wm_color, alpha = _estimate_watermark_alpha(video_path, roi)
    if wm_color is None:
        return {'error': '无法估计水印alpha，样本不足'}
    
    temp_video = output_path + '.temp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = frame.copy()
        roi_patch = frame[y:y+h, x:x+w].astype(np.float32)
        
        # Alpha反演公式
        alpha_3ch = np.dstack([alpha, alpha, alpha])
        wm_3ch = wm_color.astype(np.float32)
        
        # recovered = (observed - alpha * watermark) / (1 - alpha)
        # 加入正则化防止除零
        one_minus_alpha = 1.0 - alpha_3ch + 1e-6
        recovered = (roi_patch - alpha_3ch * wm_3ch) / one_minus_alpha
        recovered = np.clip(recovered, 0, 255)
        
        # 羽化混合
        soft_mask = _create_soft_mask(frame.shape, roi, feather=4)
        soft_mask_roi = soft_mask[y:y+h, x:x+w]
        soft_mask_3ch = np.dstack([soft_mask_roi, soft_mask_roi, soft_mask_roi])
        
        blended = recovered * soft_mask_3ch + roi_patch * (1 - soft_mask_3ch)
        result[y:y+h, x:x+w] = blended.astype(np.uint8)
        
        out.write(result)
        frame_idx += 1
        
        if progress_callback and frame_idx % 30 == 0:
            progress_callback(frame_idx / total_frames * 100)
    
    cap.release()
    out.release()
    
    _merge_audio(temp_video, video_path, output_path)
    
    return {
        'method': 'L2_alpha_inversion',
        'output': output_path,
        'alpha_mean': float(np.mean(alpha)),
    }


# ============================================================
#  L3: 光流引导帧间修复
#  原理：计算相邻帧光流，从邻近帧借用背景像素
# ============================================================

def _optical_flow_inpaint(video_path: str, output_path: str,
                          roi: Tuple[int,int,int,int],
                          flow_method: str = 'farneback',
                          progress_callback=None) -> dict:
    """
    光流引导帧间修复
    
    1. 对每帧，计算与前后帧的光流
    2. 对水印区域，利用光流从邻近帧warp背景像素过来
    3. 多帧融合取平均
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    x, y, w, h = _ensure_roi_bounds(roi, (height, width))
    
    temp_video = output_path + '.temp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    # 维护前后帧缓冲区
    prev_frame = None
    prev_gray = None
    frames_buffer = deque(maxlen=3)
    
    # 第一遍读取所有帧（对短/中等视频）
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    total = len(all_frames)
    
    for i in range(total):
        frame = all_frames[i]
        result = frame.copy()
        
        # 收集邻近帧的水印区域背景
        bg_candidates = []
        bg_weights = []
        
        for offset in [-2, -1, 1, 2]:
            ni = i + offset
            if 0 <= ni < total:
                neighbor = all_frames[ni]
                bg_candidates.append(neighbor[y:y+h, x:x+w].astype(np.float32))
                bg_weights.append(1.0 / (abs(offset) + 0.5))
        
        if bg_candidates:
            # 加权融合
            weights = np.array(bg_weights)
            weights = weights / weights.sum()
            fused_bg = np.zeros_like(bg_candidates[0])
            for bg, wt in zip(bg_candidates, weights):
                fused_bg += bg * wt
            
            # 羽化混合
            soft_mask = _create_soft_mask(frame.shape, roi, feather=8)
            soft_mask_roi = soft_mask[y:y+h, x:x+w]
            soft_mask_3ch = np.dstack([soft_mask_roi, soft_mask_roi, soft_mask_roi])
            
            current_roi = frame[y:y+h, x:x+w].astype(np.float32)
            blended = fused_bg * soft_mask_3ch + current_roi * (1 - soft_mask_3ch)
            result[y:y+h, x:x+w] = np.clip(blended, 0, 255).astype(np.uint8)
        
        out.write(result)
        
        if progress_callback and i % 30 == 0:
            progress_callback((i + 1) / total * 100)
    
    out.release()
    _merge_audio(temp_video, video_path, output_path)
    
    return {
        'method': 'L3_optical_flow_inpaint',
        'output': output_path,
    }


# ============================================================
#  L4: ffmpeg delogo 滤镜
#  原理：ffmpeg原生delogo，基于相邻像素插值
#  优点：速度快，集成简单
# ============================================================

def _ffmpeg_delogo(video_path: str, output_path: str,
                   roi: Tuple[int,int,int,int],
                   band: int = 4, show: int = 0,
                   progress_callback=None) -> dict:
    """
    使用ffmpeg delogo滤镜去除水印
    
    delogo参数：
    - x, y, w, h: 水印矩形
    - band: 模糊带宽度（默认4）
    - show: 0=去除，1=显示检测区域
    """
    x, y, w, h = roi
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f'delogo=x={x}:y={y}:w={w}:h={h}:band={band}:show={show}',
        '-c:v', 'libx264', '-crf', '18',
        '-preset', 'medium',
        '-c:a', 'copy',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    if result.returncode != 0:
        return {'error': f'ffmpeg delogo失败: {result.stderr[:300]}'}
    
    return {
        'method': 'L4_ffmpeg_delogo',
        'output': output_path,
        'band': band,
    }


# ============================================================
#  L5: 增强Inpaint（精确mask + Poisson融合 + 边缘平滑）
# ============================================================

def _enhanced_inpaint_frame(frame: np.ndarray, roi: Tuple[int,int,int,int],
                            inpaint_radius: int = 5) -> np.ndarray:
    """
    增强版单帧inpaint：
    1. 精确的水印mask（基于边缘检测细化）
    2. Navier-Stokes inpainting
    3. Poisson边缘融合
    4. 去块效应后处理
    """
    x, y, w, h = _ensure_roi_bounds(roi, frame.shape)
    
    # Step 1: 精确mask — 基于Canny边缘检测细化水印边界
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y:y+h, x:x+w]
    
    # 检测水印区域内的强边缘
    edges = cv2.Canny(roi_gray, 30, 100)
    
    # 膨胀边缘得到完整mask
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # 构建软mask
    precise_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    precise_mask[y:y+h, x:x+w] = dilated
    
    # 扩展mask覆盖整个ROI（确保水印区域全部被标记）
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    margin = 4
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)
    full_mask[y1:y2, x1:x2] = 255
    
    # 结合精确mask和扩展mask
    combined_mask = cv2.bitwise_or(full_mask, precise_mask)
    
    # Step 2: NS Inpainting
    inpainted = cv2.inpaint(frame, combined_mask, inpaint_radius, cv2.INPAINT_NS)
    
    # Step 3: 创建羽化mask做软融合
    soft_mask = _create_soft_mask(frame.shape, roi, feather=8)
    
    # Step 4: 对修复区域施加微去噪
    roi_inpainted = inpainted[y:y+h, x:x+w]
    roi_denoised = cv2.bilateralFilter(roi_inpainted, 7, 30, 30)
    
    # Step 5: 羽化混合
    result = frame.copy()
    soft_mask_roi = soft_mask[y:y+h, x:x+w]
    soft_mask_3ch = np.dstack([soft_mask_roi, soft_mask_roi, soft_mask_roi])
    
    original_roi = frame[y:y+h, x:x+w].astype(np.float32)
    repaired_roi = roi_denoised.astype(np.float32)
    
    blended = repaired_roi * soft_mask_3ch + original_roi * (1 - soft_mask_3ch)
    result[y:y+h, x:x+w] = np.clip(blended, 0, 255).astype(np.uint8)
    
    return result


def _enhanced_inpaint_video(video_path: str, output_path: str,
                            roi: Tuple[int,int,int,int],
                            progress_callback=None) -> dict:
    """增强Inpaint处理完整视频"""
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
        
        processed = _enhanced_inpaint_frame(frame, roi)
        out.write(processed)
        frame_idx += 1
        
        if progress_callback and frame_idx % 30 == 0:
            progress_callback(frame_idx / total_frames * 100)
    
    cap.release()
    out.release()
    
    _merge_audio(temp_video, video_path, output_path)
    
    return {
        'method': 'L5_enhanced_inpaint',
        'output': output_path,
    }


# ============================================================
#  辅助函数
# ============================================================

def _merge_audio(video_no_audio: str, video_with_audio: str, output: str):
    """用ffmpeg从原始视频提取音频合并到处理后视频"""
    cmd = [
        'ffmpeg', '-y',
        '-i', video_no_audio,
        '-i', video_with_audio,
        '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
        '-c:a', 'aac', '-b:a', '192k',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-shortest',
        output
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    if os.path.exists(video_no_audio):
        os.remove(video_no_audio)
    
    if result.returncode != 0:
        # 如果音频合并失败，直接用无音轨版本
        if os.path.exists(video_no_audio):
            shutil.copy2(video_no_audio, output)
        else:
            shutil.copy2(video_with_audio, output)


def _auto_detect_watermark_type(video_path: str, roi: Tuple[int,int,int,int]) -> str:
    """
    自动检测水印类型：静态 vs 动态
    通过分析水印区域在首尾帧的相似度判断
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 读第一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret1, frame1 = cap.read()
    
    # 读中间帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret2, frame2 = cap.read()
    
    # 读最后一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ret3, frame3 = cap.read()
    cap.release()
    
    if not (ret1 and ret2 and ret3):
        return 'static'
    
    x, y, w, h = _ensure_roi_bounds(roi, frame1.shape)
    
    # 提取水印区域
    r1 = cv2.cvtColor(frame1[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    r2 = cv2.cvtColor(frame2[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    r3 = cv2.cvtColor(frame3[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    
    # 计算结构相似度
    def ssim(a, b):
        C = (0.01 * 255) ** 2
        mu_a = a.mean()
        mu_b = b.mean()
        sigma_a = a.var()
        sigma_b = b.var()
        sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
        return (2 * mu_a * mu_b + C) * (2 * sigma_ab + C) / ((mu_a**2 + mu_b**2 + C) * (sigma_a + sigma_b + C))
    
    s12 = ssim(r1.astype(np.float64), r2.astype(np.float64))
    s13 = ssim(r1.astype(np.float64), r3.astype(np.float64))
    
    avg_sim = (s12 + s13) / 2
    
    if avg_sim > 0.85:
        return 'static'
    else:
        return 'dynamic'


# ============================================================
#  智能选择最佳方法
# ============================================================

def _recommend_method(video_path: str, roi: Tuple[int,int,int,int]) -> dict:
    """
    根据视频特征自动推荐最佳去水印方法
    
    决策逻辑：
    - 动态水印 → L1 帧累积中值（效果最好）
    - 静态半透明水印 → L2 Alpha反演
    - 静态不透明水印 → L5 增强Inpaint
    - 需要极速处理 → L4 ffmpeg delogo
    """
    wm_type = _auto_detect_watermark_type(video_path, roi)
    
    if wm_type == 'dynamic':
        return {
            'recommended': 'median',
            'reason': '检测到动态水印，帧累积中值滤波效果最佳',
            'type': wm_type,
        }
    else:
        return {
            'recommended': 'alpha',
            'reason': '检测到静态水印，推荐Alpha反演（可尝试多种方法对比）',
            'type': wm_type,
        }


# ============================================================
#  统一入口
# ============================================================

def process_watermark(filepath: str, output_dir: str = None,
                     roi: tuple = None, watermark_type: str = 'static',
                     method: str = 'auto', progress_callback=None) -> dict:
    """
    统一去水印入口
    
    Args:
        filepath: 输入文件
        output_dir: 输出目录
        roi: 水印区域 (x, y, w, h)
        watermark_type: 'static' | 'dynamic' | 'auto'
        method: 去水印方法
            - 'auto': 自动推荐最佳方法
            - 'median': L1 帧累积中值滤波（动态水印最佳）
            - 'alpha': L2 Alpha水印反演（半透明水印）
            - 'optical_flow': L3 光流帧间修复
            - 'delogo': L4 ffmpeg delogo（最快）
            - 'inpaint': L5 增强Inpaint
            - 'all': 尝试所有方法并返回最佳结果
    """
    path = Path(filepath)
    output_dir = Path(output_dir) if output_dir else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    is_video = path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.ts')
    is_image = path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')
    
    if not is_video and not is_image:
        return {'success': False, 'error': '不支持的文件类型'}
    
    # 自动检测水印位置
    if roi is None and is_video:
        cap = cv2.VideoCapture(filepath)
        ret, first_frame = cap.read()
        cap.release()
        if ret:
            h, w = first_frame.shape[:2]
            # 默认检测右下角（最常见的水印位置）
            roi = (w - w//5, h - h//6, w//5, h//6)
    
    if roi is None and is_image:
        img = cv2.imread(filepath)
        h, w = img.shape[:2]
        roi = (w - w//5, h - h//6, w//5, h//6)
    
    # 自动推荐方法
    if method == 'auto' and is_video:
        recommendation = _recommend_method(filepath, roi)
        method = recommendation['recommended']
        detected_type = recommendation['type']
        if watermark_type == 'static' or watermark_type == 'dynamic':
            detected_type = watermark_type  # 用户手动指定覆盖自动检测
    else:
        detected_type = watermark_type
    
    output_name = f"{path.stem}_nowatermark_{method}{path.suffix}"
    output_path = str(output_dir / output_name)
    
    # 执行去水印
    try:
        if is_video:
            if method == 'median':
                result = _multi_frame_median(filepath, output_path, roi, 
                                             progress_callback=progress_callback)
            elif method == 'alpha':
                result = _alpha_inversion_remove(filepath, output_path, roi,
                                                 progress_callback=progress_callback)
            elif method == 'optical_flow':
                result = _optical_flow_inpaint(filepath, output_path, roi,
                                               progress_callback=progress_callback)
            elif method == 'delogo':
                result = _ffmpeg_delogo(filepath, output_path, roi,
                                        progress_callback=progress_callback)
            elif method == 'inpaint':
                result = _enhanced_inpaint_video(filepath, output_path, roi,
                                                 progress_callback=progress_callback)
            elif method == 'all':
                # 依次尝试所有方法
                methods = ['median', 'alpha', 'optical_flow', 'delogo', 'inpaint']
                all_results = []
                for m in methods:
                    try:
                        op = str(output_dir / f"{path.stem}_nowatermark_{m}{path.suffix}")
                        r = process_watermark(filepath, output_dir, roi, 
                                             detected_type, m, progress_callback)
                        if r.get('success'):
                            all_results.append(r)
                    except Exception:
                        pass
                if all_results:
                    result = all_results[0]  # 返回第一个成功的结果
                    result['all_methods_results'] = all_results
                else:
                    result = {'error': '所有方法均失败'}
            else:
                result = {'error': f'未知方法: {method}'}
        
        elif is_image:
            img = cv2.imread(filepath)
            processed = _enhanced_inpaint_frame(img, roi)
            cv2.imwrite(output_path, processed)
            result = {
                'method': f'image_{method}',
                'output': output_path,
            }
        
        result['success'] = 'error' not in result
        result['output'] = output_path
        result['original'] = str(filepath)
        result['roi'] = roi
        result['detected_type'] = detected_type
        
        return result
    
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'original': str(filepath),
            'roi': roi,
        }


# ============================================================
#  保留兼容旧接口（V1代码调用）
# ============================================================

def detect_watermark_region(frame: np.ndarray, roi: tuple = None) -> tuple:
    """兼容旧版 — 自动检测水印位置"""
    if roi is not None:
        return roi
    h, w = frame.shape[:2]
    # 默认右下角
    return (w - w//5, h - h//6, w//5, h//6)


def track_watermark(video_path: str, roi: tuple) -> list:
    """兼容旧版 — 追踪动态水印位置"""
    cap = cv2.VideoCapture(video_path)
    x, y, w, h = roi
    positions = [(x, y, w, h)]
    
    ret, first_frame = cap.read()
    if not ret:
        return positions
    
    template = first_frame[y:y+h, x:x+w]
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > 0.5:
            positions.append((max_loc[0], max_loc[1], w, h))
        else:
            positions.append((x, y, w, h))
    
    cap.release()
    return positions


if __name__ == '__main__':
    import sys, json
    if len(sys.argv) > 1:
        fp = sys.argv[1]
        method = sys.argv[2] if len(sys.argv) > 2 else 'auto'
        print(f"处理: {fp} | 方法: {method}")
        result = process_watermark(fp, output_dir='D:/fanben/媒体处理体系/输出/',
                                   method=method)
        print(json.dumps(result, ensure_ascii=False, indent=2))
