"""
AI 消息示例
"""
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
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

print("1. 基本 AI 消息")
response = model.invoke("解释人工智能")
print(f"响应类型: {type(response)}")
print(f"响应内容: {response.content}")
print()

print("2. 手动创建 AI 消息用于对话历史")
# 手动创建 AI 消息 (例如，用于对话历史)
ai_msg = AIMessage("我很乐意帮助你解决这个问题！")

# 添加到对话历史
messages = [
    SystemMessage("你是一个有帮助的助手"),
    HumanMessage("你能帮我吗？"),
    ai_msg,  # 插入，就像它来自模型一样
    HumanMessage("太好了！2+2 等于多少？")
]

response = model.invoke(messages)
print(response.content)
print()

print("3. AI 消息属性")
response = model.invoke("你好")
print(f"text: {response.text}")
print(f"content: {response.content}")
print(f"id: {response.id}")
print(f"usage_metadata: {response.usage_metadata}")

