import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

'''
model = init_chat_model(
    model="qwen3-max",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
'''
model = init_chat_model(
    "openai:qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

response = model.invoke("你好")
print(response.content)