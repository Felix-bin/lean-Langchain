"""
Token 使用量示例
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

print("Token 使用量示例\n")

# 简短消息
print("1. 简短消息的 token 使用量:")
response = model.invoke("你好！")
print(f"   输入内容: 你好！")
if response.usage_metadata:
    print(f"   输入 tokens: {response.usage_metadata.get('input_tokens', 0)}")
    print(f"   输出 tokens: {response.usage_metadata.get('output_tokens', 0)}")
    print(f"   总计 tokens: {response.usage_metadata.get('total_tokens', 0)}")
print()

# 较长消息
print("2. 较长消息的 token 使用量:")
long_message = "请详细解释什么是机器学习，包括它的定义、主要应用领域和常见算法。"
response = model.invoke(long_message)
print(f"   输入内容: {long_message}")
if response.usage_metadata:
    print(f"   输入 tokens: {response.usage_metadata.get('input_tokens', 0)}")
    print(f"   输出 tokens: {response.usage_metadata.get('output_tokens', 0)}")
    print(f"   总计 tokens: {response.usage_metadata.get('total_tokens', 0)}")
print()

# 多轮对话的 token 使用
print("3. 多轮对话的 token 使用量:")
messages = [
    {"role": "system", "content": "你是一位编程助手。"},
    {"role": "user", "content": "什么是 Python？"},
    {"role": "assistant", "content": "Python 是一种高级编程语言。"},
    {"role": "user", "content": "它有什么特点？"}
]
response = model.invoke(messages)
if response.usage_metadata:
    print(f"   输入 tokens (包含对话历史): {response.usage_metadata.get('input_tokens', 0)}")
    print(f"   输出 tokens: {response.usage_metadata.get('output_tokens', 0)}")
    print(f"   总计 tokens: {response.usage_metadata.get('total_tokens', 0)}")
print()

# 完整的使用元数据
print("4. 完整的使用元数据:")
response = model.invoke("解释量子计算")
print(f"   完整元数据: {response.usage_metadata}")

