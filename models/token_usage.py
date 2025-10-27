"""
演示 Token 使用量跟踪
"""
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler, get_usage_metadata_callback
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

# 方法1: 使用回调处理器
callback = UsageMetadataCallbackHandler()
result = model.invoke("你好，请介绍一下量子计算。", config={"callbacks": [callback]})
print(f"响应内容: {result.content[:50]}...")
print(f"Token 使用情况: {callback.usage_metadata}")
print()

# 方法2: 使用上下文管理器
with get_usage_metadata_callback() as cb:
    model.invoke("你好")
    model.invoke("再见")
    print(f"总计 Token 使用情况: {cb.usage_metadata}")

