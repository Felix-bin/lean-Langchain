"""
演示 model.invoke() 的基本用法
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

# 简单调用
response = model.invoke("为什么鹦鹉有五颜六色的羽毛？")
print(response.content)

# 对话历史调用
conversation = [
    {"role": "system", "content": "你是一个有帮助的助手，负责将中文翻译成英文。"},
    {"role": "user", "content": "翻译：我喜欢编程。"},
    {"role": "assistant", "content": "I love programming."},
    {"role": "user", "content": "翻译：我喜欢构建应用程序。"}
]

response = model.invoke(conversation)
print(response.content)

