"""
多模态内容示例
注意：此示例仅展示消息结构，实际运行需要模型支持多模态功能
"""
from langchain_core.messages import HumanMessage

print("多模态消息示例\n")

# 1. 图像输入示例
print("1. 图像输入消息结构:\n")

# 从 URL
image_url_message = HumanMessage(content=[
    {"type": "text", "text": "描述这张图片的内容。"},
    {"type": "image", "url": "https://example.com/path/to/image.jpg"},
])
print("从 URL:")
print(f"  {image_url_message.content}\n")

# 从 base64 数据
image_base64_message = HumanMessage(content=[
    {"type": "text", "text": "描述这张图片的内容。"},
    {
        "type": "image",
        "base64": "iVBORw0KGgoAAAANSUhEUgAAAAUA...",
        "mime_type": "image/jpeg",
    },
])
print("从 base64:")
print(f"  类型: {image_base64_message.content[1]['type']}")
print(f"  MIME: {image_base64_message.content[1]['mime_type']}\n")

# 2. PDF 文档输入示例
print("2. PDF 文档输入消息结构:\n")

pdf_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "总结这个文档的主要内容。"},
        {"type": "file", "url": "https://example.com/document.pdf"},
    ]
}
print(f"  {pdf_message}\n")

# 3. 音频输入示例
print("3. 音频输入消息结构:\n")

audio_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "转录这段音频。"},
        {
            "type": "audio",
            "base64": "//uQxAAAAAAAAAAAAAAASW5mbw...",
            "mime_type": "audio/wav",
        },
    ]
}
print(f"  包含 {len(audio_message['content'])} 个内容块")
print(f"  音频类型: {audio_message['content'][1]['mime_type']}\n")

# 4. 视频输入示例
print("4. 视频输入消息结构:\n")

video_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这个视频的内容。"},
        {
            "type": "video",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb2...",
            "mime_type": "video/mp4",
        },
    ]
}
print(f"  视频类型: {video_message['content'][1]['mime_type']}\n")

# 5. 混合内容示例
print("5. 混合多模态内容:\n")

mixed_message = HumanMessage(content=[
    {"type": "text", "text": "分析以下内容："},
    {"type": "image", "url": "https://example.com/chart.png"},
    {"type": "text", "text": "这个图表显示了什么趋势？"},
    {"type": "file", "url": "https://example.com/data.pdf"},
])
print(f"  包含 {len(mixed_message.content)} 个内容块:")
for i, block in enumerate(mixed_message.content, 1):
    print(f"    {i}. {block['type']}")

print("\n" + "="*50)
print("\n注意: 并非所有模型都支持所有文件类型。")
print("qwen-plus 主要支持文本和图像，其他类型需要查看模型文档。")

