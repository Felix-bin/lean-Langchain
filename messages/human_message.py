"""
人类消息示例
"""
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
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

print("1. 文本内容 - 消息对象")
response = model.invoke([
    HumanMessage("什么是机器学习？")
])
print(response.content)
print()

print("2. 文本内容 - 字符串快捷方式")
# 使用字符串是单个 HumanMessage 的快捷方式
response = model.invoke("什么是机器学习？")
print(response.content)
print()

print("3. 带元数据的消息")
human_msg = HumanMessage(
    content="你好！我想了解深度学习。",
    name="alice",  # 可选：识别不同用户
    id="msg_123",  # 可选：用于追踪的唯一标识符
)
print(f"消息内容: {human_msg.content}")
print(f"用户名: {human_msg.name}")
print(f"消息ID: {human_msg.id}")

response = model.invoke([human_msg])
print(f"\n模型响应: {response.content}")

