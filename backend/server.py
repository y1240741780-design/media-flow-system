"""
媒体处理体系 — Flask Web服务器
整合 MD5修改、去水印、文案生成、在线下载 四大模块
"""

import os
import sys
import json
import uuid
import shutil
import threading
from pathlib import Path
from datetime import datetime

# 添加backend到路径
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask import render_template_string
from werkzeug.utils import secure_filename

from md5_changer import process_md5, get_md5, get_file_info
from watermark_remover import process_watermark
from copywriter import generate_copywriting
from downloader import download_media, batch_download, get_platform_info

app = Flask(__name__, static_folder='../web/static', static_url_path='/static')

# 配置
BASE_DIR = Path('D:/fanben/媒体处理体系')
UPLOAD_DIR = BASE_DIR / '输入'
OUTPUT_DIR = BASE_DIR / '输出'
ALLOWED_EXTENSIONS = {
    'video': {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'},
    'image': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'},
}
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 处理任务状态存储
tasks = {}


def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in (ALLOWED_EXTENSIONS['video'] | ALLOWED_EXTENSIONS['image'])


# ============ 首页 ============

@app.route('/')
def index():
    """返回前端主页"""
    html_path = BASE_DIR / 'web' / 'index.html'
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    return '<h1>媒体处理体系</h1><p>前端页面缺失</p>'


# ============ 文件上传 ============

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传文件"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    files = request.files.getlist('file')
    results = []
    
    for file in files:
        if file.filename == '':
            continue
        
        if not allowed_file(file.filename):
            results.append({'filename': file.filename, 'error': '不支持的文件类型'})
            continue
        
        filename = secure_filename(file.filename)
        # 添加UUID避免重名
        unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        filepath = UPLOAD_DIR / unique_name
        file.save(str(filepath))
        
        info = get_file_info(str(filepath))
        results.append(info)
    
    return jsonify({'files': results})


# ============ MD5修改 ============

@app.route('/api/md5/change', methods=['POST'])
def api_md5_change():
    """MD5修改"""
    data = request.get_json()
    filepath = data.get('filepath', '')
    methods = data.get('methods', ['append'])
    quality = data.get('quality', 'medium')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': '文件不存在'}), 400
    
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {'status': 'processing', 'type': 'md5'}
    
    results = process_md5(filepath, methods=methods,
                         output_dir=str(OUTPUT_DIR), quality=quality)
    
    tasks[task_id] = {'status': 'completed', 'type': 'md5', 'results': results}
    
    return jsonify({
        'task_id': task_id,
        'results': results,
        'original_md5': get_md5(filepath),
    })


# ============ 去水印 ============

@app.route('/api/watermark/preview', methods=['POST'])
def api_watermark_preview():
    """获取视频/图片预览帧，用于框选水印"""
    import cv2
    import base64
    
    data = request.get_json()
    filepath = data.get('filepath', '')
    frame_index = data.get('frame_index', 0)
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': '文件不存在'}), 400
    
    info = get_file_info(filepath)
    
    if info['is_image']:
        img = cv2.imread(filepath)
        _, buffer = cv2.imencode('.jpg', img)
        b64 = base64.b64encode(buffer).decode()
        return jsonify({
            'type': 'image',
            'width': img.shape[1],
            'height': img.shape[0],
            'data': f'data:image/jpeg;base64,{b64}',
        })
    
    elif info['is_video']:
        cap = cv2.VideoCapture(filepath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 跳到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'error': '无法读取视频帧'}), 400
        
        _, buffer = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(buffer).decode()
        
        return jsonify({
            'type': 'video',
            'width': frame.shape[1],
            'height': frame.shape[0],
            'data': f'data:image/jpeg;base64,{b64}',
            'total_frames': total_frames,
            'fps': fps,
        })
    
    return jsonify({'error': '不支持的文件'}), 400


@app.route('/api/watermark/remove', methods=['POST'])
def api_watermark_remove():
    """执行去水印"""
    data = request.get_json()
    filepath = data.get('filepath', '')
    roi = tuple(data.get('roi')) if data.get('roi') else None
    watermark_type = data.get('watermark_type', 'static')
    method = data.get('method', 'inpaint')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': '文件不存在'}), 400
    
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {'status': 'processing', 'type': 'watermark'}
    
    try:
        result = process_watermark(
            filepath,
            output_dir=str(OUTPUT_DIR),
            roi=roi,
            watermark_type=watermark_type,
            method=method,
        )
        tasks[task_id] = {'status': 'completed', 'type': 'watermark', 'results': result}
        return jsonify({'task_id': task_id, **result})
    except Exception as e:
        tasks[task_id] = {'status': 'error', 'type': 'watermark', 'error': str(e)}
        return jsonify({'error': str(e)}), 500


# ============ 文案生成 ============

@app.route('/api/copywriting/generate', methods=['POST'])
def api_copywriting():
    """生成文案"""
    data = request.get_json()
    filepath = data.get('filepath', '')
    options = data.get('options', {})
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': '文件不存在'}), 400
    
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {'status': 'processing', 'type': 'copywriting'}
    
    try:
        result = generate_copywriting(filepath, options)
        tasks[task_id] = {'status': 'completed', 'type': 'copywriting', 'results': result}
        return jsonify({'task_id': task_id, **result})
    except Exception as e:
        tasks[task_id] = {'status': 'error', 'type': 'copywriting', 'error': str(e)}
        return jsonify({'error': str(e)}), 500


# ============ 在线下载 ============

@app.route('/api/download', methods=['POST'])
def api_download():
    """下载在线视频"""
    data = request.get_json()
    url = data.get('url', '')
    options = data.get('options', {})
    
    if not url:
        return jsonify({'error': 'URL不能为空'}), 400
    
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {'status': 'processing', 'type': 'download'}
    
    result = download_media(url, str(OUTPUT_DIR), options=options)
    tasks[task_id] = {'status': 'completed', 'type': 'download', 'results': result}
    
    return jsonify({'task_id': task_id, **result})


@app.route('/api/download/batch', methods=['POST'])
def api_batch_download():
    """批量下载"""
    data = request.get_json()
    urls = data.get('urls', [])
    options = data.get('options', {})
    
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {'status': 'processing', 'type': 'batch_download'}
    
    results = batch_download(urls, str(OUTPUT_DIR), options=options)
    tasks[task_id] = {'status': 'completed', 'type': 'batch_download', 'results': results}
    
    return jsonify({'task_id': task_id, 'results': results})


@app.route('/api/platforms', methods=['GET'])
def api_platforms():
    """获取支持的平台列表"""
    return jsonify(get_platform_info())


# ============ 文件管理 ============

@app.route('/api/files', methods=['GET'])
def api_list_files():
    """列出输出目录中的文件"""
    files_dir = request.args.get('dir', 'output')
    target_dir = OUTPUT_DIR if files_dir == 'output' else UPLOAD_DIR
    
    files = []
    for f in sorted(target_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.is_file():
            info = get_file_info(str(f))
            files.append(info)
    
    return jsonify({'files': files, 'dir': str(target_dir)})


@app.route('/api/files/download/<path:filename>', methods=['GET'])
def api_download_file(filename):
    """下载处理后的文件"""
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        return jsonify({'error': '文件不存在'}), 404
    return send_file(str(filepath), as_attachment=True)


# ============ 任务状态 ============

@app.route('/api/task/<task_id>', methods=['GET'])
def api_task_status(task_id):
    """查询任务状态"""
    task = tasks.get(task_id)
    if task:
        return jsonify({'task_id': task_id, **task})
    return jsonify({'error': '任务不存在'}), 404


# ============ 启动服务器 ============

if __name__ == '__main__':
    print("=" * 50)
    print("  媒体处理体系 Web 服务器启动中...")
    print(f"  上传目录: {UPLOAD_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  打开浏览器访问: http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
