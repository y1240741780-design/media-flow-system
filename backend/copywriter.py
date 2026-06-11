"""
文案生成模块 — 基于视频信息生成文案
支持：模板模式（离线） + AI增强（需API Key）
"""

import os
import json
import random
import subprocess
from pathlib import Path
from datetime import datetime


# ============ 视频信息提取 ============

def get_video_metadata(filepath: str) -> dict:
    """使用ffprobe提取视频元数据"""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams',
        filepath
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        
        format_info = data.get('format', {})
        video_stream = None
        audio_stream = None
        
        for stream in data.get('streams', []):
            if stream['codec_type'] == 'video' and video_stream is None:
                video_stream = stream
            elif stream['codec_type'] == 'audio' and audio_stream is None:
                audio_stream = stream
        
        return {
            'filename': os.path.basename(filepath),
            'duration': float(format_info.get('duration', 0)),
            'duration_str': format_duration(float(format_info.get('duration', 0))),
            'size_mb': round(float(format_info.get('size', 0)) / (1024 * 1024), 2),
            'format': format_info.get('format_name', ''),
            'bitrate': int(format_info.get('bit_rate', 0)) // 1000 if format_info.get('bit_rate') else 0,
            'width': video_stream.get('width', 0) if video_stream else 0,
            'height': video_stream.get('height', 0) if video_stream else 0,
            'codec': video_stream.get('codec_name', '') if video_stream else '',
            'fps': eval(video_stream.get('r_frame_rate', '0/1')) if video_stream else 0,
            'has_audio': audio_stream is not None,
            'audio_codec': audio_stream.get('codec_name', '') if audio_stream else '',
            'orientation': '竖屏' if video_stream and video_stream.get('width', 0) < video_stream.get('height', 0) else '横屏',
        }
    except Exception as e:
        return {
            'filename': os.path.basename(filepath),
            'error': str(e),
            'duration': 0,
            'duration_str': '未知',
            'size_mb': 0,
            'width': 0,
            'height': 0,
        }


def format_duration(seconds: float) -> str:
    """格式化时长"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}分{s}秒"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}时{m}分{s}秒"


# ============ 模板生成 ============

# 文案模板库
TITLE_TEMPLATES = {
    '短视频通用': [
        "【必看】{topic}，看完你会感谢我！",
        "99%的人不知道的{topic}秘密！",
        "{topic}，这个方法太绝了！",
        "我用了这个方法，{benefit}立竿见影！",
        "学会这招，{topic}变得超简单！",
        "偷偷告诉你{topic}的小技巧～",
        "{topic}原来这么简单？很多人第一步就错了",
        "别再乱{topic}了，试试这个方法",
    ],
    '教程类': [
        "新手必学：{topic}完整教程",
        "从零开始学{topic}，这个视频就够了",
        "手把手教你{topic}，包教包会",
        "{topic}教程 | 一个视频全搞定",
        "3分钟学会{topic}，超详细步骤",
    ],
    '产品展示': [
        "这个{product}真的太好用了！",
        "{product}开箱体验，说说真实感受",
        "用了{product}一个月，我后悔了吗？",
        "性价比超高的{product}推荐",
    ],
    '生活记录': [
        "记录{scene}的美好时刻 🌟",
        "今天{scene}，太开心了！",
        "平凡的一天，因为{scene}而不同",
    ],
}

DESC_TEMPLATES = [
    "📌 视频时长：{duration}\n📷 分辨率：{width}x{height}\n\n{body}\n\n#️⃣ {tags}",
    "✨ {body}\n\n---\n⏱ {duration} | 🎬 {resolution}\n\n{tags}", 
    "{body}\n\n💡 关注我，每天分享更多精彩内容！\n\n{tags}",
]

HASHTAG_POOL = {
    '教程': ['教程', '干货', '学习', '技能', '技巧', '方法', '指南'],
    '生活': ['日常', '生活', 'vlog', '记录', '分享', '美好'],
    '技术': ['技术', '科技', '数码', '软件', '工具', '效率'],
    '搞笑': ['搞笑', '沙雕', '快乐', '幽默', '有趣'],
    '美食': ['美食', '吃货', '做饭', '食谱', '料理'],
    '旅游': ['旅游', '旅行', '风景', '打卡', '户外'],
    '通用': ['热门', '推荐', '必看', '涨知识', 'get新技能'],
}


def generate_title(metadata: dict, style: str = '短视频通用') -> str:
    """生成标题"""
    templates = TITLE_TEMPLATES.get(style, TITLE_TEMPLATES['短视频通用'])
    template = random.choice(templates)
    
    # 从文件名推断主题
    topic = metadata.get('filename', '这个视频').rsplit('.', 1)[0]
    # 清理文件名中的特殊字符
    topic = topic.replace('-', ' ').replace('_', ' ').strip()
    if len(topic) > 15:
        topic = topic[:15] + '...'
    
    return template.format(
        topic=topic,
        benefit=random.choice(['效果', '变化', '结果', '体验']),
        product=topic,
        scene=topic,
    )


def generate_description(metadata: dict, body: str = '', tags: list = None) -> str:
    """生成描述文案"""
    template = random.choice(DESC_TEMPLATES)
    
    if tags is None:
        tags = generate_tags(metadata, 5)
    
    return template.format(
        duration=metadata.get('duration_str', '未知'),
        width=metadata.get('width', 0),
        height=metadata.get('height', 0),
        resolution=f"{metadata.get('width', 0)}x{metadata.get('height', 0)}",
        body=body or f"这个视频展示了{metadata.get('filename', '')}的精彩内容",
        tags=' '.join(f"#{t}" for t in tags),
    )


def generate_tags(metadata: dict, count: int = 5) -> list:
    """生成标签"""
    # 根据视频属性选择标签池
    orientation = metadata.get('orientation', '横屏')
    duration = metadata.get('duration', 0)
    
    category = '通用'
    if duration < 30:
        category = '搞笑'
    elif duration < 180:
        category = '生活'
    elif duration > 600:
        category = '教程'
    
    pool = HASHTAG_POOL.get(category, HASHTAG_POOL['通用'])
    
    # 从文件名提取关键词
    name = metadata.get('filename', '').rsplit('.', 1)[0]
    words = [w for w in name.replace('-', ' ').replace('_', ' ').split() if len(w) > 1]
    
    tags = words[:3] + random.sample(pool, min(count, len(pool)))
    
    # 去重
    seen = set()
    unique_tags = []
    for t in tags:
        if t.lower() not in seen:
            unique_tags.append(t)
            seen.add(t.lower())
    
    return unique_tags[:count]


def generate_article(metadata: dict, topic: str = '') -> str:
    """生成配套文章/脚本"""
    duration_str = metadata.get('duration_str', '未知')
    resolution = f"{metadata.get('width', 0)}x{metadata.get('height', 0)}"
    name = metadata.get('filename', '视频').rsplit('.', 1)[0]
    
    return f"""【{name}】视频解说脚本

## 基本信息
- 视频时长：{duration_str}
- 分辨率：{resolution}
- 画面方向：{metadata.get('orientation', '横屏')}

## 开场白
大家好！今天给大家带来一个关于{topic or name}的精彩分享。

## 内容要点
1. 首先我们看到的是...（此处根据视频实际内容填写）
2. 接着展示了...（此处根据视频实际内容填写）
3. 最精彩的部分是...（此处根据视频实际内容填写）

## 结尾总结
希望这个视频对你有帮助！如果喜欢，记得点赞关注，我们下期再见！

## SEO信息
- 标题建议：{generate_title(metadata)}
- 标签建议：{' '.join(f'#{t}' for t in generate_tags(metadata))}
- 适用平台：抖音 / 快手 / 小红书 / B站 / YouTube
"""


def generate_copywriting(filepath: str, options: dict = None) -> dict:
    """
    统一入口：生成文案
    
    Args:
        filepath: 视频文件路径
        options: {
            'type': 'all' | 'title' | 'description' | 'tags' | 'article',
            'style': '短视频通用' | '教程类' | '产品展示' | '生活记录',
            'use_ai': False,
            'ai_api_key': '',
            'ai_api_url': '',
            'ai_model': '',
        }
    """
    options = options or {}
    metadata = get_video_metadata(filepath)
    
    if 'error' in metadata:
        return {'success': False, 'error': metadata['error']}
    
    gen_type = options.get('type', 'all')
    style = options.get('style', '短视频通用')
    use_ai = options.get('use_ai', False)
    
    result = {
        'success': True,
        'metadata': metadata,
    }
    
    if gen_type in ('all', 'title'):
        result['title'] = generate_title(metadata, style)
    
    if gen_type in ('all', 'description'):
        tags = generate_tags(metadata)
        result['description'] = generate_description(metadata, tags=tags)
    
    if gen_type in ('all', 'tags'):
        result['tags'] = generate_tags(metadata, 10)
    
    if gen_type in ('all', 'article'):
        topic = metadata.get('filename', '').rsplit('.', 1)[0]
        result['article'] = generate_article(metadata, topic)
    
    # AI增强（如果启用且有API Key）
    if use_ai and options.get('ai_api_key'):
        ai_result = ai_enhance_copywriting(metadata, result, options)
        if ai_result.get('success'):
            result['ai_enhanced'] = ai_result
    
    return result


def ai_enhance_copywriting(metadata: dict, base_result: dict, options: dict) -> dict:
    """
    AI增强文案生成
    支持 OpenAI兼容API（包括DeepSeek等）
    """
    import urllib.request
    import urllib.error
    
    api_key = options.get('ai_api_key', os.environ.get('OPENAI_API_KEY', ''))
    api_url = options.get('ai_api_url', 'https://api.openai.com/v1/chat/completions')
    model = options.get('ai_model', 'gpt-3.5-turbo')
    
    if not api_key:
        return {'success': False, 'error': '未配置API Key'}
    
    prompt = f"""你是一个专业的内容创作者。请根据以下视频信息，生成吸引人的文案。

视频信息：
- 文件名：{metadata.get('filename', '')}
- 时长：{metadata.get('duration_str', '')}
- 分辨率：{metadata.get('width', 0)}x{metadata.get('height', 0)}
- 画面方向：{metadata.get('orientation', '')}

请返回JSON格式：
{{
  "title": "吸引人的标题（20字以内）",
  "description": "视频描述（100字以内）",
  "tags": ["标签1", "标签2", "标签3", "标签4", "标签5"],
  "hook": "视频开头黄金3秒的引导语"
}}

注意：
- 标题要有吸引力，带emoji
- 描述要简洁有力
- 标签要精准热门
- 只用JSON格式回复
"""
    
    data = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.8,
    }).encode()
    
    req = urllib.request.Request(api_url, data=data)
    req.add_header('Content-Type', 'application/json')
    req.add_header('Authorization', f'Bearer {api_key}')
    
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        body = json.loads(resp.read())
        content = body['choices'][0]['message']['content']
        
        # 尝试提取JSON
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]
        
        ai_result = json.loads(content.strip())
        return {'success': True, **ai_result}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        fp = sys.argv[1]
        result = generate_copywriting(fp)
        print(json.dumps(result, ensure_ascii=False, indent=2))
