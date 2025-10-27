# 消息 (Messages)

消息是 LangChain 中模型上下文的基本单位。它们代表模型的输入和输出，携带表示与 LLM 交互时对话状态所需的内容和元数据。

消息对象包含:

- [**角色 (Role)**](#message-types) - 标识消息类型 (例如 `system`, `user`)
- [**内容 (Content)**](#message-content) - 表示消息的实际内容 (如文本、图像、音频、文档等)
- [**元数据 (Metadata)**](#message-metadata) - 可选字段，如响应信息、消息 ID 和 token 使用量

LangChain 提供了一个跨所有模型提供商都能工作的标准消息类型，确保无论调用哪个模型都具有一致的行为。

## 基本用法 (Basic usage)

使用消息最简单的方式是创建消息对象并在[调用](/oss/python/langchain/models#invocation)时将它们传递给模型。

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

system_msg = SystemMessage("你是一个有帮助的助手。")
human_msg = HumanMessage("你好，你好吗？")

# 与聊天模型一起使用
messages = [system_msg, human_msg]
response = model.invoke(messages)  # 返回 AIMessage
```

### 文本提示 (Text prompts)

文本提示是字符串 - 适合直接生成任务，不需要保留对话历史。

```python
response = model.invoke("写一首关于春天的诗")
```

**何时使用文本提示：**

- 只有一个独立的请求
- 不需要对话历史
- 希望代码复杂度最小

### 消息提示 (Message prompts)

或者，您可以通过提供消息对象列表来向模型传递消息列表。

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("你是一位诗歌专家"),
    HumanMessage("写一首关于春天的诗"),
    AIMessage("春风拂面花开...")
]
response = model.invoke(messages)
```

**何时使用消息提示：**

- 管理多轮对话
- 处理多模态内容 (图像、音频、文件)
- 包含系统指令

### 字典格式 (Dictionary format)

您还可以直接使用 OpenAI 聊天完成格式指定消息。

```python
messages = [
    {"role": "system", "content": "你是一位诗歌专家"},
    {"role": "user", "content": "写一首关于春天的诗"},
    {"role": "assistant", "content": "春风拂面花开..."}
]
response = model.invoke(messages)
```

## 消息类型 (Message types)

- [系统消息 (System message)](#system-message) - 告诉模型如何行为并为交互提供上下文
- [人类消息 (Human message)](#human-message) - 表示用户输入和与模型的交互
- [AI 消息 (AI message)](#ai-message) - 模型生成的响应，包括文本内容、工具调用和元数据
- [工具消息 (Tool message)](#tool-message) - 表示[工具调用](/oss/python/langchain/models#tool-calling)的输出

### 系统消息 (System Message)

[`SystemMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.SystemMessage) 表示一组初始指令，用于引导模型的行为。您可以使用系统消息来设定语气、定义模型的角色，并为响应建立指南。

**基本指令:**

```python
from langchain_core.messages import SystemMessage, HumanMessage

system_msg = SystemMessage("你是一个有帮助的编程助手。")

messages = [
    system_msg,
    HumanMessage("如何创建 REST API？")
]
response = model.invoke(messages)
```

**详细角色设定:**

```python
from langchain_core.messages import SystemMessage, HumanMessage

system_msg = SystemMessage("""
你是一位资深 Python 开发者，擅长 Web 框架。
始终提供代码示例并解释你的推理过程。
在解释中做到简洁但全面。
""")

messages = [
    system_msg,
    HumanMessage("如何创建 REST API？")
]
response = model.invoke(messages)
```

---

### 人类消息 (Human Message)

[`HumanMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.HumanMessage) 表示用户输入和交互。它们可以包含文本、图像、音频、文件和任何其他多模态[内容](#message-content)。

#### 文本内容

**消息对象:**

```python
from langchain_core.messages import HumanMessage

response = model.invoke([
    HumanMessage("什么是机器学习？")
])
```

**字符串快捷方式:**

```python
# 使用字符串是单个 HumanMessage 的快捷方式
response = model.invoke("什么是机器学习？")
```

#### 消息元数据

```python
human_msg = HumanMessage(
    content="你好！",
    name="alice",  # 可选：识别不同用户
    id="msg_123",  # 可选：用于追踪的唯一标识符
)
```

> **注意**: `name` 字段的行为因提供商而异 - 有些用它来识别用户，有些则忽略它。要检查，请参考模型提供商的[参考文档](https://reference.langchain.com/python/integrations/)。

---

### AI 消息 (AI Message)

[`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 表示模型调用的输出。它们可以包含多模态数据、工具调用以及您稍后可以访问的提供商特定元数据。

```python
response = model.invoke("解释 AI")
print(type(response))  # <class 'langchain_core.messages.AIMessage'>
```

模型在调用时会返回 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 对象，其中包含响应中的所有关联元数据。

提供商对不同类型的消息进行权重/上下文化的方式不同，这意味着有时手动创建一个新的 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 对象并将其插入消息历史中（就像它来自模型一样）会很有帮助。

```python
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# 手动创建 AI 消息 (例如，用于对话历史)
ai_msg = AIMessage("我很乐意帮助你解决这个问题！")

# 添加到对话历史
messages = [
    SystemMessage("你是一个有帮助的助手"),
    HumanMessage("你能帮我吗？"),
    ai_msg,  # 插入，就像它来自模型一样
    HumanMessage("太好了！2+2 等于多少？")
]

response = model.invoke(messages)
```

**属性说明:**

- **`text`** (string) - 消息的文本内容
- **`content`** (string | dict[]) - 消息的原始内容
- **`content_blocks`** (ContentBlock[]) - 消息的标准化[内容块](#message-content)
- **`tool_calls`** (dict[] | None) - 模型进行的工具调用。如果没有调用工具则为空
- **`id`** (string) - 消息的唯一标识符 (由 LangChain 自动生成或在提供商响应中返回)
- **`usage_metadata`** (dict | None) - 消息的使用元数据，可在可用时包含 token 计数
- **`response_metadata`** (ResponseMetadata | None) - 消息的响应元数据

#### 工具调用 (Tool calls)

当模型进行[工具调用](/oss/python/langchain/models#tool-calling)时，它们会包含在 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 中：

```python
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

def get_weather(location: str) -> str:
    """获取某个位置的天气。"""
    ...

model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("巴黎的天气怎么样？")

for tool_call in response.tool_calls:
    print(f"工具: {tool_call['name']}")
    print(f"参数: {tool_call['args']}")
    print(f"ID: {tool_call['id']}")
```

其他结构化数据，如推理或引用，也可以出现在消息[内容](/oss/python/langchain/messages#message-content)中。

#### Token 使用量 (Token usage)

[`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 可以在其 [`usage_metadata`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage.usage_metadata) 字段中保存 token 计数和其他使用元数据：

```python
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

response = model.invoke("你好！")
print(response.usage_metadata)
```

输出示例:

```python
{
    'input_tokens': 2,
    'output_tokens': 15,
    'total_tokens': 17
}
```

详见 [`UsageMetadata`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage.usage_metadata)。

#### 流式传输和块 (Streaming and chunks)

在流式传输期间，您将收到 [`AIMessageChunk`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessageChunk) 对象，这些对象可以组合成完整的消息对象：

```python
chunks = []
full_message = None
for chunk in model.stream("你好"):
    chunks.append(chunk)
    print(chunk.content)
    full_message = chunk if full_message is None else full_message + chunk
```

> **💡 了解更多**:
>
> - [从聊天模型流式传输 token](/oss/python/langchain/models#stream)
> - [从智能体流式传输 token 和/或步骤](/oss/python/langchain/streaming)

---

### 工具消息 (Tool Message)

对于支持[工具调用](/oss/python/langchain/models#tool-calling)的模型，AI 消息可以包含工具调用。工具消息用于将单个工具执行的结果传递回模型。

[工具](/oss/python/langchain/tools)可以直接生成 [`ToolMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ToolMessage) 对象。下面我们展示一个简单的示例。在[工具指南](/oss/python/langchain/tools)中阅读更多。

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# 模型进行工具调用后
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "北京"},
        "id": "call_123"
    }]
)

# 执行工具并创建结果消息
weather_result = "晴天，22°C"
tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # 必须匹配调用 ID
)

# 继续对话
messages = [
    HumanMessage("北京的天气怎么样？"),
    ai_message,  # 模型的工具调用
    tool_message,  # 工具执行结果
]
response = model.invoke(messages)  # 模型处理结果
```

**属性说明:**

- **`content`** (string, 必需) - 工具调用的字符串化输出
- **`tool_call_id`** (string, 必需) - 此消息响应的工具调用的 ID (必须与 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 中的工具调用 ID 匹配)
- **`name`** (string, 必需) - 被调用的工具名称
- **`artifact`** (dict) - 不发送给模型但可以通过编程方式访问的附加数据

> **注意**: `artifact` 字段存储不会发送给模型但可以通过编程方式访问的补充数据。这对于存储原始结果、调试信息或用于下游处理的数据很有用，而不会使模型的上下文变得混乱。
>
> **示例：为检索元数据使用 artifact**
>
> 例如，一个[检索](/oss/python/langchain/retrieval)工具可以从文档中检索一段文本供模型参考。消息 `content` 包含模型将参考的文本，`artifact` 可以包含应用程序可以使用的文档标识符或其他元数据 (例如，渲染页面)。示例如下：
>
> ```python
> from langchain_core.messages import ToolMessage
>
> # 发送给模型
> message_content = "这是最好的时代，也是最坏的时代。"
>
> # Artifact 可在下游使用
> artifact = {"document_id": "doc_123", "page": 0}
>
> tool_message = ToolMessage(
>     content=message_content,
>     tool_call_id="call_123",
>     name="search_books",
>     artifact=artifact,
> )
> ```
>
> 参见 [RAG 教程](/oss/python/langchain/rag) 以获取使用 LangChain 构建检索[智能体](/oss/python/langchain/agents)的端到端示例。

---

## 消息内容 (Message content)

您可以将消息的内容视为发送给模型的数据有效载荷。消息有一个 `content` 属性，它是松散类型的，支持字符串和无类型对象列表 (例如字典)。这允许在 LangChain 聊天模型中直接支持提供商原生结构，例如[多模态](#multimodal)内容和其他数据。

另外，LangChain 为文本、推理、引用、多模态数据、服务器端工具调用和其他消息内容提供专用的内容类型。请参阅下面的[内容块](#standard-content-blocks)。

LangChain 聊天模型接受 `content` 属性中的消息内容，可以包含：

1. 字符串
2. 提供商原生格式的内容块列表
3. [LangChain 标准内容块](#standard-content-blocks)列表

以下是使用[多模态](#multimodal)输入的示例：

```python
from langchain_core.messages import HumanMessage

# 字符串内容
human_message = HumanMessage("你好，你好吗？")

# 提供商原生格式 (例如 OpenAI)
human_message = HumanMessage(content=[
    {"type": "text", "text": "你好，你好吗？"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

# 标准内容块列表
human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "你好，你好吗？"},
    {"type": "image", "url": "https://example.com/image.jpg"},
])
```

> **💡 提示**: 初始化消息时指定 `content_blocks` 仍会填充消息 `content`，但为这样做提供了类型安全的接口。

### 标准内容块 (Standard content blocks)

LangChain 提供了一个跨提供商工作的消息内容标准表示。

消息对象实现了一个 `content_blocks` 属性，它会延迟解析 `content` 属性为标准的、类型安全的表示。例如，从 [ChatAnthropic](/oss/python/integrations/chat/anthropic) 或 [ChatOpenAI](/oss/python/integrations/chat/openai) 生成的消息将包含各自提供商格式的 `thinking` 或 `reasoning` 块，但可以被延迟解析为一致的 [`ReasoningContentBlock`](#content-block-reference) 表示。

参见[集成指南](/oss/python/integrations/providers/overview)以开始使用您选择的推理提供商。

> **注意**: **序列化标准内容**
>
> 如果 LangChain 之外的应用程序需要访问标准内容块表示，您可以选择在消息内容中存储内容块。
>
> 为此，您可以将 `LC_OUTPUT_VERSION` 环境变量设置为 `v1`。或者，使用 `output_version="v1"` 初始化任何聊天模型：
>
> ```python
> from langchain.chat_models import init_chat_model
>
> model = init_chat_model("openai:qwen-plus", output_version="v1")
> ```

### 多模态 (Multimodal)

**多模态**是指能够处理不同形式的数据的能力，例如文本、音频、图像和视频。LangChain 为这些数据提供了标准类型，可以跨提供商使用。

[聊天模型](/oss/python/langchain/models)可以接受多模态数据作为输入并将其作为输出生成。下面我们展示了包含多模态数据的输入消息的简短示例。

> **注意**: 额外的键可以包含在内容块的顶级，或嵌套在 `"extras": {"key": value}` 中。
>
> 例如，[OpenAI](/oss/python/integrations/chat/openai#pdfs) 和 [AWS Bedrock Converse](/oss/python/integrations/chat/bedrock) 需要 PDF 的文件名。有关详细信息，请参阅所选模型的[提供商页面](/oss/python/integrations/providers/overview)。

**图像输入:**

```python
# 从 URL
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这张图片的内容。"},
        {"type": "image", "url": "https://example.com/path/to/image.jpg"},
    ]
}

# 从 base64 数据
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这张图片的内容。"},
        {
            "type": "image",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "image/jpeg",
        },
    ]
}

# 从提供商管理的文件 ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这张图片的内容。"},
        {"type": "image", "file_id": "file-abc123"},
    ]
}
```

**PDF 文档输入:**

```python
# 从 URL
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这个文档的内容。"},
        {"type": "file", "url": "https://example.com/path/to/document.pdf"},
    ]
}

# 从 base64 数据
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这个文档的内容。"},
        {
            "type": "file",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "application/pdf",
        },
    ]
}

# 从提供商管理的文件 ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这个文档的内容。"},
        {"type": "file", "file_id": "file-abc123"},
    ]
}
```

**音频输入:**

```python
# 从 base64 数据
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这个音频的内容。"},
        {
            "type": "audio",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "audio/wav",
        },
    ]
}

# 从提供商管理的文件 ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这个音频的内容。"},
        {"type": "audio", "file_id": "file-abc123"},
    ]
}
```

**视频输入:**

```python
# 从 base64 数据
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这个视频的内容。"},
        {
            "type": "video",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "video/mp4",
        },
    ]
}

# 从提供商管理的文件 ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述这个视频的内容。"},
        {"type": "video", "file_id": "file-abc123"},
    ]
}
```

> **警告**: 并非所有模型都支持所有文件类型。请查看模型提供商的[参考文档](https://reference.langchain.com/python/integrations/)以了解支持的格式和大小限制。

### 内容块参考 (Content block reference)

内容块表示为类型化字典列表 (在创建消息或访问 `content_blocks` 属性时)。列表中的每个项目必须遵循以下块类型之一：

#### 核心类型 (Core)

**TextContentBlock (文本内容块)**

用途：标准文本输出

- **`type`** (string, 必需): 始终为 `"text"`
- **`text`** (string, 必需): 文本内容
- **`annotations`** (object[]): 文本的注释列表
- **`extras`** (object): 附加的提供商特定数据

示例:

```python
{
    "type": "text",
    "text": "你好世界",
    "annotations": []
}
```

**ReasoningContentBlock (推理内容块)**

用途：模型推理步骤

- **`type`** (string, 必需): 始终为 `"reasoning"`
- **`reasoning`** (string): 推理内容
- **`extras`** (object): 附加的提供商特定数据

示例:

```python
{
    "type": "reasoning",
    "reasoning": "用户正在询问...",
    "extras": {"signature": "abc123"},
}
```

#### 多模态类型 (Multimodal)

**ImageContentBlock (图像内容块)**

用途：图像数据

- **`type`** (string, 必需): 始终为 `"image"`
- **`url`** (string): 指向图像位置的 URL
- **`base64`** (string): Base64 编码的图像数据
- **`id`** (string): 外部存储图像的引用 ID
- **`mime_type`** (string): 图像 [MIME 类型](https://www.iana.org/assignments/media-types/media-types.xhtml#image) (例如 `image/jpeg`, `image/png`)

**AudioContentBlock (音频内容块)**

用途：音频数据

- **`type`** (string, 必需): 始终为 `"audio"`
- **`url`** (string): 指向音频位置的 URL
- **`base64`** (string): Base64 编码的音频数据
- **`id`** (string): 外部存储音频文件的引用 ID
- **`mime_type`** (string): 音频 [MIME 类型](https://www.iana.org/assignments/media-types/media-types.xhtml#audio) (例如 `audio/mpeg`, `audio/wav`)

**VideoContentBlock (视频内容块)**

用途：视频数据

- **`type`** (string, 必需): 始终为 `"video"`
- **`url`** (string): 指向视频位置的 URL
- **`base64`** (string): Base64 编码的视频数据
- **`id`** (string): 外部存储视频文件的引用 ID
- **`mime_type`** (string): 视频 [MIME 类型](https://www.iana.org/assignments/media-types/media-types.xhtml#video) (例如 `video/mp4`, `video/webm`)

**FileContentBlock (文件内容块)**

用途：通用文件 (PDF 等)

- **`type`** (string, 必需): 始终为 `"file"`
- **`url`** (string): 指向文件位置的 URL
- **`base64`** (string): Base64 编码的文件数据
- **`id`** (string): 外部存储文件的引用 ID
- **`mime_type`** (string): 文件 [MIME 类型](https://www.iana.org/assignments/media-types/media-types.xhtml) (例如 `application/pdf`)

**PlainTextContentBlock (纯文本内容块)**

用途：文档文本 (`.txt`, `.md`)

- **`type`** (string, 必需): 始终为 `"text-plain"`
- **`text`** (string): 文本内容
- **`mime_type`** (string): 文本的 [MIME 类型](https://www.iana.org/assignments/media-types/media-types.xhtml) (例如 `text/plain`, `text/markdown`)

#### 工具调用类型 (Tool Calling)

**ToolCall (工具调用)**

用途：函数调用

- **`type`** (string, 必需): 始终为 `"tool_call"`
- **`name`** (string, 必需): 要调用的工具名称
- **`args`** (object, 必需): 传递给工具的参数
- **`id`** (string, 必需): 此工具调用的唯一标识符

示例:

```python
{
    "type": "tool_call",
    "name": "search",
    "args": {"query": "天气"},
    "id": "call_123"
}
```

**ToolCallChunk (工具调用块)**

用途：流式工具调用片段

- **`type`** (string, 必需): 始终为 `"tool_call_chunk"`
- **`name`** (string): 被调用的工具名称
- **`args`** (string): 部分工具参数 (可能是不完整的 JSON)
- **`id`** (string): 工具调用标识符
- **`index`** (number | string): 此块在流中的位置

**InvalidToolCall (无效工具调用)**

用途：格式错误的调用，用于捕获 JSON 解析错误

- **`type`** (string, 必需): 始终为 `"invalid_tool_call"`
- **`name`** (string): 调用失败的工具名称
- **`args`** (object): 传递给工具的参数
- **`error`** (string): 出错描述

#### 服务器端工具执行 (Server-Side Tool Execution)

**ServerToolCall (服务器工具调用)**

用途：服务器端执行的工具调用

- **`type`** (string, 必需): 始终为 `"server_tool_call"`
- **`id`** (string, 必需): 与工具调用关联的标识符
- **`name`** (string, 必需): 要调用的工具名称
- **`args`** (string, 必需): 部分工具参数 (可能是不完整的 JSON)

**ServerToolCallChunk (服务器工具调用块)**

用途：流式服务器端工具调用片段

- **`type`** (string, 必需): 始终为 `"server_tool_call_chunk"`
- **`id`** (string): 与工具调用关联的标识符
- **`name`** (string): 被调用的工具名称
- **`args`** (string): 部分工具参数 (可能是不完整的 JSON)
- **`index`** (number | string): 此块在流中的位置

**ServerToolResult (服务器工具结果)**

用途：搜索结果

- **`type`** (string, 必需): 始终为 `"server_tool_result"`
- **`tool_call_id`** (string, 必需): 对应服务器工具调用的标识符
- **`id`** (string): 与服务器工具结果关联的标识符
- **`status`** (string, 必需): 服务器端工具的执行状态。`"success"` 或 `"error"`
- **`output`**: 执行的工具的输出

#### 提供商特定块 (Provider-Specific Blocks)

**NonStandardContentBlock (非标准内容块)**

用途：提供商特定的逃生舱

- **`type`** (string, 必需): 始终为 `"non_standard"`
- **`value`** (object, 必需): 提供商特定的数据结构

用法：用于实验性或提供商独有的功能

其他提供商特定的内容类型可以在每个模型提供商的[参考文档](/oss/python/integrations/providers/overview)中找到。

> **提示**: 在 [API 参考](https://reference.langchain.com/python/langchain/messages)中查看规范类型定义。

> **ℹ信息**: 内容块作为消息上的新属性在 LangChain v1 中引入，以标准化跨提供商的内容格式，同时保持与现有代码的向后兼容性。内容块不是 [`content`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.messages.BaseMessage.content) 属性的替代品，而是可用于以标准化格式访问消息内容的新属性。

## 与聊天模型一起使用 (Use with chat models)

[聊天模型](/oss/python/langchain/models)接受消息对象序列作为输入，并返回 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 作为输出。交互通常是无状态的，因此简单的对话循环涉及使用不断增长的消息列表调用模型。

参考以下指南以了解更多：

- [持久化和管理对话历史](/oss/python/langchain/short-term-memory)的内置功能
- 管理上下文窗口的策略，包括[修剪和总结消息](/oss/python/langchain/short-term-memory#common-patterns)

---

> **提示**: 在 [API 参考](https://reference.langchain.com/python/langchain/messages)中查看完整的类型定义。
