"""
工具调用示例
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

def get_weather(location: str) -> str:
    """获取某个位置的天气。"""
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，22°C",
        "上海": "多云，25°C",
        "广州": "雨天，28°C",
        "巴黎": "阴天，18°C",
    }
    return weather_data.get(location, f"{location}的天气未知")

print("工具调用示例")
model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("北京的天气怎么样？")

print("模型响应:")
print(f"内容: {response.content}")
print()

if response.tool_calls:
    print("工具调用信息:")
    for tool_call in response.tool_calls:
        print(f"  工具: {tool_call['name']}")
        print(f"  参数: {tool_call['args']}")
        print(f"  ID: {tool_call['id']}")
        print()
        
        # 执行工具调用
        if tool_call['name'] == 'get_weather':
            location = tool_call['args'].get('location', '')
            result = get_weather(location)
            print(f"  工具执行结果: {result}")
else:
    print("没有工具调用")

