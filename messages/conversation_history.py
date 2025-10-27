"""
对话历史管理示例
"""
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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


# 初始化对话历史
conversation_history = [
    SystemMessage("你是一个有帮助的编程助手。")
]

print("开始对话...\n")

# 第一轮对话
print("用户: 什么是 Python？")
conversation_history.append(HumanMessage("什么是 Python？"))

response = model.invoke(conversation_history)
print(f"AI: {response.content}\n")
conversation_history.append(response)

# 第二轮对话 - 引用之前的上下文
print("用户: 它的主要特点是什么？")
conversation_history.append(HumanMessage("它的主要特点是什么？"))

response = model.invoke(conversation_history)
print(f"AI: {response.content}\n")
conversation_history.append(response)

# 第三轮对话 - 继续引用上下文
print("用户: 给我一个简单的代码示例")
conversation_history.append(HumanMessage("给我一个简单的代码示例"))

response = model.invoke(conversation_history)
print(f"AI: {response.content}\n")
conversation_history.append(response)

# 显示完整对话历史
print("\n完整对话历史:")
for i, msg in enumerate(conversation_history, 1):
    role = msg.__class__.__name__
    content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
    print(f"{i}. {role}: {content}")

print(f"\n对话历史中共有 {len(conversation_history)} 条消息")

