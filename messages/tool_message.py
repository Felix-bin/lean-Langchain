"""
工具消息示例 - 完整的工具调用流程
"""
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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

print("完整的工具调用流程\n")

# 定义工具
def get_weather(location: str) -> str:
    """获取某个位置的天气。"""
    weather_data = {
        "北京": "晴天，22°C",
        "上海": "多云，25°C",
    }
    return weather_data.get(location, f"{location}的天气未知")

# 步骤1: 用户提问
user_question = "北京的天气怎么样？"
print(f"用户: {user_question}\n")

# 步骤2: 绑定工具并调用模型
model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke(user_question)

# 步骤3: 检查是否有工具调用
if response.tool_calls:
    tool_call = response.tool_calls[0]
    print(f"模型决定调用工具: {tool_call['name']}")
    print(f"工具参数: {tool_call['args']}\n")
    
    # 步骤4: 执行工具
    location = tool_call['args'].get('location', '')
    weather_result = get_weather(location)
    print(f"工具执行结果: {weather_result}\n")
    
    # 步骤5: 创建工具消息
    tool_message = ToolMessage(
        content=weather_result,
        tool_call_id=tool_call['id'],
        name="get_weather"
    )
    
    # 步骤6: 将工具结果返回给模型
    messages = [
        HumanMessage(user_question),
        response,  # 包含工具调用的 AI 消息
        tool_message,  # 工具执行结果
    ]
    
    final_response = model.invoke(messages)
    print(f"最终回答: {final_response.content}")
else:
    print(f"直接回答: {response.content}")

print("\n" + "="*50 + "\n")

# 使用 artifact 的示例
print("使用 artifact 存储额外信息\n")

# 模拟一个返回文档的工具
message_content = "这是最好的时代，也是最坏的时代。"
artifact = {"document_id": "doc_123", "page": 0, "source": "双城记"}

tool_message = ToolMessage(
    content=message_content,
    tool_call_id="call_456",
    name="search_books",
    artifact=artifact,
)

print(f"消息内容 (发送给模型): {tool_message.content}")
print(f"Artifact (不发送给模型): {tool_message.artifact}")
print(f"  - 文档ID: {tool_message.artifact['document_id']}")
print(f"  - 页码: {tool_message.artifact['page']}")
print(f"  - 来源: {tool_message.artifact['source']}")

