"""
演示结构化输出的用法
"""
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
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

# 定义数据结构
class Movie(BaseModel):
    """电影信息"""
    title: str = Field(..., description="电影标题")
    year: int = Field(..., description="上映年份")
    director: str = Field(..., description="导演姓名")
    rating: float = Field(..., description="电影评分(满分10分)")

# 使用结构化输出
# 注意：qwen 模型要求提示词中必须包含 "json" 字样
model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke(
    "请以 JSON 格式提供电影《盗梦空间》的详细信息，包括标题、年份、导演和评分"
)

print(f"标题: {response.title}")
print(f"年份: {response.year}")
print(f"导演: {response.director}")
print(f"评分: {response.rating}")
print()

# 使用 include_raw 获取原始消息
print("=== 使用 include_raw ===")
model_with_raw = model.with_structured_output(Movie, include_raw=True)
response_with_raw = model_with_raw.invoke(
    "请以 JSON 格式提供电影《星际穿越》的详细信息，包括标题、年份、导演和评分"
)

print("解析的数据:")
print(f"  标题: {response_with_raw['parsed'].title}")
print(f"  年份: {response_with_raw['parsed'].year}")
print(f"  导演: {response_with_raw['parsed'].director}")
print(f"  评分: {response_with_raw['parsed'].rating}")
print()
if response_with_raw['raw'].usage_metadata:
    print(f"Token 使用信息: {response_with_raw['raw'].usage_metadata}")

