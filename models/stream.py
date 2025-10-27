"""
演示 model.stream() 流式传输的用法
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

# 基本流式传输
for chunk in model.stream("为什么鹦鹉有五颜六色的羽毛？"):
    print(chunk.content, end="", flush=True)
print("\n")

# 累积流式内容
full = None  # None | AIMessageChunk
for chunk in model.stream("天空是什么颜色？"):
    full = chunk if full is None else full + chunk

#print(f"完整内容: {full.content}")

