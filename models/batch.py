"""
演示 model.batch() 批处理的用法
"""
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

# 初始化模型
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# 批量处理
responses = model.batch([
    "为什么鹦鹉有五颜六色的羽毛？",
    "飞机是如何飞行的？",
    "什么是量子计算？"
])

for i, response in enumerate(responses, 1):
    print(f"\n问题 {i} 的回答:")
    print(response.content)

