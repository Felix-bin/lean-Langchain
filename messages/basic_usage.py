"""
消息的基本用法示例
"""
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

# 使用消息对象
system_msg = SystemMessage("你是一个有帮助的助手。")
human_msg = HumanMessage("你好，你好吗？")

messages = [system_msg, human_msg]
response = model.invoke(messages)
print(f"响应类型: {type(response)}")
print(f"响应内容: {response.content}")
print()

#使用文本提示
response = model.invoke("写一首关于春天的诗")
print(response.content)
print()

# 使用消息列表
messages = [
    SystemMessage("你是一位诗歌专家"),
    HumanMessage("写一首关于春天的诗"),
    AIMessage("春风拂面花开...")
]
response = model.invoke(messages)
print(response.content)
print()

# 使用字典格式
messages = [
    {"role": "system", "content": "你是一位诗歌专家"},
    {"role": "user", "content": "写一首关于春天的诗"},
    {"role": "assistant", "content": "春风拂面花开..."}
]
response = model.invoke(messages)
print(response.content)

