from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

chatLLM = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/
)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你是谁？"}]

# 使用 stream() 方法实现流式输出
for chunk in chatLLM.stream(messages):
    print(chunk.content, end="", flush=True)
