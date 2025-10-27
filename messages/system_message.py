"""
系统消息示例
"""
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
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

system_msg = SystemMessage("你是一个有帮助的编程助手。")

messages = [
    system_msg,
    HumanMessage("如何创建 REST API？")
]
response = model.invoke(messages)
print(response.content)
print("\n" + "="*50 + "\n")

system_msg = SystemMessage("""
你是一位资深 Python 开发者，擅长 Web 框架。
始终提供代码示例并解释你的推理过程。
在解释中做到简洁但全面。
""")

messages = [
    system_msg,
    HumanMessage("如何使用 FastAPI 创建一个简单的 REST API？")
]
response = model.invoke(messages)
print(response.content)

