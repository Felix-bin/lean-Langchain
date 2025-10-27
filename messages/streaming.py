"""
流式传输和消息块示例
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


print("1. 基本流式传输:")
print("AI: ", end="", flush=True)
for chunk in model.stream("用一句话介绍机器学习"):
    print(chunk.content, end="", flush=True)
print("\n")

print("2. 收集所有块:")
chunks = []
for chunk in model.stream("什么是深度学习？"):
    chunks.append(chunk)
print(f"   收到 {len(chunks)} 个块")
print(f"   第一个块: {chunks[0].content if chunks else 'N/A'}")
print()

print("3. 累积完整消息:")
full_message = None
for chunk in model.stream("解释神经网络"):
    full_message = chunk if full_message is None else full_message + chunk

print(f"   完整消息类型: {type(full_message)}")
print(f"   完整消息内容: {full_message.content}")
print()

print("4. 流式传输时显示进度:")
print("AI: ", end="", flush=True)
chunk_count = 0
for chunk in model.stream("写一首关于春天的短诗"):
    print(chunk.content, end="", flush=True)
    chunk_count += 1
print(f"\n   (共 {chunk_count} 个块)")

