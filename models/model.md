# 模型 (Models)

[大语言模型 (LLMs)](https://en.wikipedia.org/wiki/Large_language_model) 是强大的 AI 工具，能够像人类一样理解和生成文本。它们功能多样，足以完成内容创作、语言翻译、文本摘要和问答等任务，而无需针对每个任务进行专门训练。

除了文本生成之外，许多模型还支持以下功能：

- 🔨 [工具调用 (Tool calling)](#tool-calling) - 调用外部工具（如数据库查询或 API 调用）并在响应中使用结果。

  - **详细说明：** 工具调用让模型能够突破纯文本生成的限制，与外部系统交互。例如，当你问"今天北京的天气如何？"时，模型可以调用天气 API 获取实时数据，而不是基于训练数据猜测。这大大扩展了模型的实用性。

- 🔷 [结构化输出 (Structured output)](#structured-outputs) - 模型的响应被约束为遵循定义的格式。

  - **详细说明：** 结构化输出确保模型返回的数据符合特定的格式规范（如 JSON Schema、Pydantic 模型等）。这对于需要将模型输出直接集成到应用程序中的场景至关重要，可以避免解析错误和数据验证问题。

- 🖼️ [多模态 (Multimodality)](#multimodal) - 处理和返回文本以外的数据，如图像、音频和视频。

  - **详细说明：** 多模态模型可以理解和生成多种类型的数据。例如，它们可以分析图片内容、生成图像、转录语音或理解视频场景。这使得 AI 应用能够处理更丰富的现实世界数据。

- 🧠 [推理 (Reasoning)](#reasoning) - 模型执行多步推理以得出结论。
  - **详细说明：** 推理能力使模型能够像人类一样进行逻辑思考。模型会将复杂问题分解为多个步骤，逐步推导，而不是直接给出答案。这对于解决数学问题、逻辑谜题或需要深度分析的任务特别有用。

Models 是 [智能体 (agents)](/oss/python/langchain/agents) 的推理引擎。它们驱动智能体的决策过程，决定调用哪些工具、如何解释结果以及何时提供最终答案。

LangChain 提供了统一的抽象层，这意味着您可以使用相同的代码与不同提供商的模型交互。这种设计带来了巨大的灵活性——您可以在开发时使用廉价的模型，在生产环境切换到更强大的模型，而无需重写代码。

## 基本用法 (Basic usage)

模型可以通过两种方式使用：

init_chat_model 和 Model Class

### 初始化模型 (Initialize a model)

在 LangChain 中开始使用独立模型的最简单方法是使用 [`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model) 从您选择的[提供商](/oss/python/integrations/providers/overview)初始化一个模型（示例如下）：

**使用 init_chat_model:**

```python
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="qwen3-max",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

response = model.invoke("你好")
print(response.content)
```

**使用 Model Class:**

```python
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

chatLLM = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)

response = chatLLM.invoke(“你好”)
print(response.content)
```

两种方式本质上是相同的 - init_chat_model 最终也会实例化对应的 Model Class。选择哪种方式主要取决于:

- 灵活性需求: 如果需要动态切换模型 → init_chat_model

- 明确性需求: 如果模型确定且需要类型安全 → Model Class

- 你的场景: 对于 DashScope 这种兼容 OpenAI API 的服务,两种方式都可以,但 init_chat_model 更简洁

### 核心方法 (Key methods)

#### 📤 [Invoke（调用)](#invoke)

模型接受消息作为输入，在生成完整响应后输出消息。

**详细说明：** 这是最基本的同步调用方式，适合需要完整响应后再继续执行的场景。

#### 📡 [Stream（流式传输）](#stream)

调用模型，但在生成输出时实时流式传输。

**详细说明：** 流式传输显著改善用户体验，特别是对于长文本生成。用户可以立即看到输出开始出现，而不是等待整个响应完成。

#### 📊 [Batch（批处理）](#batch)

批量发送多个请求到模型以实现更高效的处理。

**详细说明：** 批处理可以并行处理多个独立请求，大大提高吞吐量和效率，特别适合处理大量数据的场景。

## 参数 (Parameters)

聊天模型接受可用于配置其行为的参数。支持的完整参数集因模型和提供商而异，但标准参数包括：

**详细说明：** 参数配置是调优模型行为的关键。通过调整这些参数，您可以在创造性、确定性、成本和性能之间找到平衡。例如，较高的 `temperature` 会产生更有创意但可能不太一致的输出，而较低的值则产生更可预测的结果。

- **model** (string, 必需)

  要与提供商一起使用的特定模型的名称或标识符。

  **详细说明：** 例如 "gpt-4"、"claude-3-5-sonnet" 等。不同的模型有不同的能力、速度和成本特征。

- **api_key** (string)

  与模型提供商进行身份验证所需的密钥。通常在注册访问模型时颁发。通常通过设置环境变量来访问（环境变量：一个其值在程序外部设置的变量，通常通过操作系统或微服务的内置功能）。

  **详细说明：** API 密钥用于身份验证和计费。永远不要在代码中硬编码 API 密钥，应该使用环境变量（如 `OPENAI_API_KEY`）或密钥管理服务。这样可以保护密钥不被意外泄露到版本控制系统中。

- **temperature** (number)

  控制模型输出的随机性。较高的数字使响应更具创造性；较低的数字使响应更具确定性。

  **详细说明：** `temperature` 通常在 0 到 2 之间（具体范围取决于提供商）：

  - **0-0.3**: 高度确定性，适合事实性任务、代码生成、数据提取
  - **0.7-1.0**: 平衡创造性和一致性，适合通用对话
  - **1.0+**: 高度创造性，适合创意写作、头脑风暴

- **timeout** (number)

  在取消请求之前等待模型响应的最长时间（以秒为单位）。

  **详细说明：** 超时设置可以防止应用程序因模型响应缓慢而挂起。对于生产环境，建议设置合理的超时值（如 30-60 秒）。如果经常超时，可能需要考虑使用更快的模型或优化提示。

- **max_tokens** (number)

  限制响应中的 token 总数，有效控制输出的长度。（Token：模型读取和生成的基本单位。提供商可能有不同的定义，但通常它们可以表示一个完整的单词或单词的一部分。）

  **详细说明：** Token 是模型处理文本的基本单位。对于英文，1 个 token 约等于 0.75 个单词；对于中文，1 个汉字通常是 1-2 个 token。设置 `max_tokens` 可以：

  - 控制成本（大多数提供商按 token 计费）
  - 防止过长的响应
  - 确保响应适合您的 UI 限制

  注意：输入和输出 token 都计入模型的上下文窗口限制。

- **max_retries** (number)

  如果由于网络超时或速率限制等问题导致请求失败，系统将尝试重新发送请求的最大次数。

  **详细说明：** 自动重试机制可以提高应用程序的可靠性，特别是在处理瞬时网络问题或 API 速率限制时。LangChain 会使用指数退避策略，在重试之间逐渐增加等待时间。

使用 [`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model),将这些参数作为内联 `**kwargs`(任意关键字参数,更多信息见 [Python args kwargs](https://www.w3schools.com/python/python_args_kwargs.asp))传递:

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
    # Kwargs passed to the model:
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)
```

> **ℹ️ 提示**: 每个聊天模型集成可能有额外的参数用于控制提供商特定的功能。例如,[`ChatOpenAI`](https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI/) 有 `use_responses_api` 来指定是否使用 OpenAI Responses 或 Completions API。要查找给定聊天模型支持的所有参数,请访问[聊天模型集成](/oss/python/integrations/chat)页面。

---

## 调用 (Invocation)

必须调用聊天模型才能生成输出。有三种主要的调用方法，每种都适合不同的使用场景。

**详细说明：** 选择合适的调用方法对应用程序的性能和用户体验至关重要：

- **Invoke**: 适合需要完整响应的简单场景
- **Stream**: 适合交互式应用，提供实时反馈
- **Batch**: 适合处理大量独立请求的场景

### Invoke（调用）

调用模型最直接的方法是使用 [`invoke()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.invoke) 配合单个消息或消息列表。

**详细说明：** `invoke()` 是一个阻塞调用，意味着您的程序会等待模型完成整个响应后再继续。这对于批处理脚本或不需要实时反馈的场景很理想。

```python
response = model.invoke("为什么鹦鹉有五颜六色的羽毛？")
print(response.content)
```

可以向模型提供消息列表来表示对话历史。每条消息都有一个角色，模型使用该角色来指示在对话中谁发送了该消息。有关角色、类型和内容的更多详细信息，请参阅[消息指南](/oss/python/langchain/messages)。

**详细说明：** 消息历史对于构建对话式应用至关重要。通过提供上下文，模型可以：

- 记住之前的对话内容
- 理解代词引用（如"它"、"那个"）
- 保持对话的连贯性和一致性
- 遵循多轮对话中建立的指令

消息角色包括：

- **system**: 设置模型的行为和角色（如"你是一个有帮助的助手"）
- **user**: 用户的输入或问题
- **assistant**: 模型之前的响应
- **tool**: 工具调用的结果

**字典格式:**

```python
conversation = [
    {"role": "system", "content": "你是一个有帮助的助手，负责将中文翻译成英文。"},
    {"role": "user", "content": "翻译：我喜欢编程。"},
    {"role": "assistant", "content": "I love programming."},
    {"role": "user", "content": "翻译：我喜欢构建应用程序。"}
]

response = model.invoke(conversation)
print(response.content)  # I love building applications.
```

**消息对象格式:**

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
    SystemMessage("你是一个有帮助的助手，负责将中文翻译成英文。"),
    HumanMessage("翻译：我喜欢编程。"),
    AIMessage("I love programming."),
    HumanMessage("翻译：我喜欢构建应用程序。")
]

response = model.invoke(conversation)
print(response.content)  # I love building applications.
```

### Stream（流式传输）

大多数模型可以在生成输出内容时进行流式传输。通过逐步显示输出，流式传输显著改善了用户体验，特别是对于较长的响应。

**详细说明：** 流式传输的优势：

1. **更好的用户体验**: 用户立即看到响应开始出现，而不是盯着加载动画等待
2. **感知性能提升**: 即使总时间相同，流式显示让应用感觉更快
3. **早期错误检测**: 可以在生成过程中检测问题，而不是等到最后
4. **适合长文本**: 对于生成长文档或代码，流式传输是必不可少的

实际应用场景：

- 聊天机器人界面（类似 ChatGPT 的逐字显示）
- 代码生成工具
- 长文本摘要或翻译
- 实时内容创作助手

调用 [`stream()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.stream) 返回一个<Tooltip tip="一个对象，按顺序逐步提供对集合中每个项的访问。">迭代器</Tooltip>，它会在生成输出块时产出它们。您可以使用循环实时处理每个块：

**详细说明：** 迭代器模式使得流式传输在 Python 中非常自然。每次循环迭代都会等待下一个块到达，然后立即处理它。这种方式内存效率高，因为不需要一次性存储整个响应。

**基本文本流式传输:**

```python
for chunk in model.stream("为什么鹦鹉有五颜六色的羽毛？"):
    print(chunk.content, end="", flush=True)
```

**流式传输工具调用、推理和其他内容:**

```python
for chunk in model.stream("天空是什么颜色？"):
    # 获取完整内容
    print(chunk.content, end="", flush=True)
```

As opposed to [`invoke()`](#invoke), which returns a single [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) after the model has finished generating its full response, `stream()` returns multiple [`AIMessageChunk`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessageChunk) objects, each containing a portion of the output text. Importantly, each chunk in a stream is designed to be gathered into a full message via summation:

```python
full = None  # None | AIMessageChunk
for chunk in model.stream("天空是什么颜色？"):
    full = chunk if full is None else full + chunk
    print(full.content)

# 天
# 天空
# 天空是
# 天空是蓝
# 天空是蓝色
# 天空是蓝色的
# ...

print(full.content)
# 天空是蓝色的...
```

The resulting message can be treated the same as a message that was generated with [`invoke()`](#invoke) - for example, it can be aggregated into a message history and passed back to the model as conversational context.

> **⚠️ 警告**: 流式传输仅在程序中的所有步骤都知道如何处理块流时才有效。例如,不具备流式传输能力的应用程序是需要在处理之前将整个输出存储在内存中的应用程序。

#### 高级流式传输主题

##### "自动流式传输"聊天模型

LangChain 通过在某些情况下自动启用流式传输模式来简化从聊天模型进行流式传输,即使您没有显式调用流式传输方法。当您使用非流式传输的 invoke 方法但仍然希望流式传输整个应用程序(包括来自聊天模型的中间结果)时,这特别有用。

例如,在 [LangGraph agents](/oss/python/langchain/agents) 中,您可以在节点内调用 `model.invoke()`,但如果在流式传输模式下运行,LangChain 将自动委托给流式传输。

**工作原理**

当您 `invoke()` 一个聊天模型时,如果 LangChain 检测到您正在尝试流式传输整个应用程序,它将自动切换到内部流式传输模式。就使用 invoke 的代码而言,调用的结果将是相同的;然而,在聊天模型被流式传输时,LangChain 将负责在 LangChain 的回调系统中调用 [`on_llm_new_token`](https://reference.langchain.com/python/langchain_core/callbacks/#langchain_core.callbacks.base.AsyncCallbackHandler.on_llm_new_token) 事件。

回调事件允许 LangGraph `stream()` 和 [`astream_events()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.astream_events) 实时显示聊天模型的输出。

##### 流式传输事件

LangChain 聊天模型还可以使用 [`astream_events()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.astream_events) 流式传输语义事件。

这简化了基于事件类型和其他元数据的过滤,并将在后台聚合完整消息。请参阅下面的示例。

```python
async for event in model.astream_events("你好"):

    if event["event"] == "on_chat_model_start":
        print(f"Input: {event['data']['input']}")

    elif event["event"] == "on_chat_model_stream":
        print(f"Token: {event['data']['chunk'].content}")

    elif event["event"] == "on_chat_model_end":
        print(f"Full message: {event['data']['output'].content}")

    else:
        pass
```

输出示例:

```txt
Input: 你好
Token: 你
Token: 好
Token: ！
Token: 有
Token: 什
Token: 么
...
Full message: 你好！有什么可以帮助你的吗？
```

> **💡 提示**: 有关事件类型和其他详细信息,请参阅 [`astream_events()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.astream_events) 参考文档。

### Batch

Batching a collection of independent requests to a model can significantly improve performance and reduce costs, as the processing can be done in parallel:

```python
responses = model.batch([
    "为什么鹦鹉有五颜六色的羽毛？",
    "飞机是如何飞行的？",
    "什么是量子计算？"
])
for response in responses:
    print(response.content)
```

> **📝 注意**: 本节描述的是聊天模型方法 [`batch()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch),它在客户端并行化模型调用。这与推理提供商支持的批处理 API **不同**,例如 [OpenAI](https://platform.openai.com/docs/guides/batch) 或 [Anthropic](https://docs.claude.com/en/docs/build-with-claude/batch-processing#message-batches-api)。

By default, [`batch()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch) will only return the final output for the entire batch. If you want to receive the output for each individual input as it finishes generating, you can stream results with [`batch_as_completed()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch_as_completed):

```python
for response in model.batch_as_completed([
    "为什么鹦鹉有五颜六色的羽毛？",
    "飞机是如何飞行的？",
    "什么是量子计算？"
]):
    print(response)
```

> **📝 注意**: 使用 [`batch_as_completed()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch_as_completed) 时,结果可能会乱序到达。每个结果都包含输入索引,以便在需要时匹配并重建原始顺序。

> **💡 提示**: 使用 [`batch()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch) 或 [`batch_as_completed()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.batch_as_completed) 处理大量输入时,您可能希望控制并行调用的最大数量。可以通过在 [`RunnableConfig`](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.RunnableConfig) 字典中设置 [`max_concurrency`](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.RunnableConfig.max_concurrency) 属性来实现。
>
> ```python
> model.batch(
>     list_of_inputs,
>     config={
>         'max_concurrency': 5,  # Limit to 5 parallel calls
>     }
> )
> ```
>
> 有关支持的所有属性的完整列表,请参阅 [`RunnableConfig`](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.RunnableConfig) 参考文档。

## 结构化输出 (Structured outputs)

可以请求模型以符合给定模式的格式提供响应。这对于确保输出可以轻松解析并用于后续处理非常有用。LangChain 支持多种模式类型和强制执行结构化输出的方法。

**详细说明：** 结构化输出解决了 LLM 输出解析的难题。没有结构化输出时，您需要：

1. 在提示中描述期望的格式（可能不可靠）
2. 编写复杂的解析逻辑来提取信息
3. 处理各种格式错误和边缘情况
4. 验证提取的数据

使用结构化输出：

- **保证格式正确**: 模型的输出自动符合定义的模式
- **类型安全**: 使用 Python 类型提示，获得 IDE 自动完成和类型检查
- **自动验证**: Pydantic 等工具自动验证数据
- **易于集成**: 输出可以直接用作函数参数或保存到数据库

实际应用场景：

- **数据提取**: 从文本中提取结构化信息（姓名、日期、地址等）
- **分类**: 将文本分类到预定义类别
- **表单填充**: 从自然语言生成表单数据
- **API 集成**: 生成符合 API 要求的请求数据
- **数据转换**: 将一种格式的数据转换为另一种格式

#### 使用 Pydantic

[Pydantic models](https://docs.pydantic.dev/latest/concepts/models/#basic-model-usage) 提供了最丰富的功能集,包括字段验证、描述和嵌套结构。

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("提供电影《盗梦空间》的详细信息")
print(response)  # Movie(title="盗梦空间", year=2010, director="克里斯托弗·诺兰", rating=8.8)
```

#### 使用 JSON Schema

为了获得最大的控制或互操作性,您可以提供原始 JSON Schema。

```python
import json

json_schema = {
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "The title of the movie"
        },
        "year": {
            "type": "integer",
            "description": "The year the movie was released"
        },
        "director": {
            "type": "string",
            "description": "The director of the movie"
        },
        "rating": {
            "type": "number",
            "description": "The movie's rating out of 10"
        }
    },
    "required": ["title", "year", "director", "rating"]
}

model_with_structure = model.with_structured_output(
    json_schema,
    method="json_schema",
)
response = model_with_structure.invoke("提供电影《盗梦空间》的详细信息")
print(response)  # {'title': '盗梦空间', 'year': 2010, ...}
```

> **📝 注意**:
>
> **结构化输出的关键考虑因素:**
>
> - **Method 参数**: 一些提供商支持不同的方法 (`'json_schema'`, `'function_calling'`, `'json_mode'`)
>   - `'json_schema'` 通常是指提供商提供的专用结构化输出功能
>   - `'function_calling'` 通过强制遵循给定模式的[工具调用](#tool-calling)来派生结构化输出
>   - `'json_mode'` 是一些提供商提供的 `'json_schema'` 的前身 - 它生成有效的 json,但必须在提示中描述模式
> - **Include raw**: 使用 `include_raw=True` 可以同时获取解析的输出和原始 AI 消息
> - **Validation**: Pydantic 模型提供自动验证,而 `TypedDict` 和 JSON Schema 需要手动验证

#### 示例:解析结构和消息输出

It can be useful to return the raw [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) object alongside the parsed representation to access response metadata such as [token counts](#token-usage). To do this, set [`include_raw=True`](<https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.with_structured_output(include_raw)>) when calling [`with_structured_output`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.with_structured_output):

```python theme={null}
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie, include_raw=True)  # [!code highlight]
response = model_with_structure.invoke("提供电影《盗梦空间》的详细信息")
response
# {
#     "raw": AIMessage(...),
#     "parsed": Movie(title=..., year=..., ...),
#     "parsing_error": None,
# }
```

## 高级主题 (Advanced topics)

### 多模态 (Multimodal)

某些模型可以处理和返回非文本数据，如图像、音频和视频。您可以通过提供[内容块](/oss/python/langchain/messages#message-content)向模型传递非文本数据。

**详细说明：** 多模态能力使 AI 应用能够处理真实世界的丰富数据。这开启了许多新的应用场景：

**视觉理解**:

- 图像分析和描述
- 文档理解（扫描件、发票、表格）
- 图表和图形解读
- OCR（光学字符识别）
- 视觉问答

**音频处理**:

- 语音转文本
- 音频内容理解
- 多语言语音识别

**多模态输出**:

- 图像生成（文本到图像）
- 图像编辑和修改
- 图表和可视化创建

实际应用示例：

- 智能文档处理系统（理解扫描的合同、发票）
- 医学影像分析助手
- 教育应用（解释教科书图表）
- 客户服务机器人（理解用户上传的截图）
- 内容审核系统

> **💡 提示**: 所有具有底层多模态功能的 LangChain 聊天模型都支持:
>
> 1. 跨提供商标准格式的数据（参见[我们的消息指南](/oss/python/langchain/messages)）
> 2. OpenAI [聊天完成](https://platform.openai.com/docs/api-reference/chat)格式
> 3. 特定提供商原生的任何格式（例如，Anthropic 模型接受 Anthropic 原生格式）

See the [multimodal section](/oss/python/langchain/messages#multimodal) of the messages guide for details.

[某些模型](https://models.dev/)可以将多模态数据作为其响应的一部分返回（注意:并非所有 LLM 都是平等的!）。如果被调用这样做,生成的 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 将具有多模态类型的内容块。

```python
response = model.invoke("创建一张猫的图片")
print(response.content)
# 注意：qwen-plus 不支持图像生成，这里仅作示例
# 支持多模态的模型会返回包含图像数据的响应
```

See the [integrations page](/oss/python/integrations/providers/overview) for details on specific providers.

### 推理 (Reasoning)

较新的模型能够执行多步推理以得出结论。这涉及将复杂问题分解为更小、更易管理的步骤。

**详细说明：** 推理能力代表了 LLM 的重大进步。传统模型倾向于直接给出答案，而推理模型会：

**工作方式**:

1. **分析问题**: 理解问题的各个组成部分
2. **制定计划**: 确定解决问题的步骤
3. **逐步求解**: 执行每个步骤，使用中间结果
4. **验证答案**: 检查结果是否合理
5. **提供最终答案**: 给出经过验证的结论

**优势**:

- **更高准确性**: 特别是对于需要多步逻辑的问题
- **可解释性**: 您可以看到模型的"思考过程"
- **错误诊断**: 当答案错误时，可以定位问题出在哪一步
- **可靠性**: 减少"幻觉"（编造信息）的情况

**适用场景**:

- 数学问题求解
- 逻辑推理和谜题
- 代码调试和错误分析
- 复杂决策制定
- 科学问题分析
- 法律案例分析

**如果底层模型支持，** 您可以展示这个推理过程，以更好地理解模型如何得出最终答案。

**详细说明：** 访问推理过程不仅有助于调试和验证，还能用于教育场景（展示解题步骤）或增强用户对 AI 决策的信任。

**流式输出推理过程:**

```python
for chunk in model.stream("为什么鹦鹉有五颜六色的羽毛？"):
    print(chunk.content, end="", flush=True)
```

**完整推理输出:**

```python
response = model.invoke("为什么鹦鹉有五颜六色的羽毛？")
print(response.content)
```

Depending on the model, you can sometimes specify the level of effort it should put into reasoning. Similarly, you can request that the model turn off reasoning entirely. This may take the form of categorical "tiers" of reasoning (e.g., `'low'` or `'high'`) or integer token budgets.

For details, see the [integrations page](/oss/python/integrations/providers/overview) or [reference](https://reference.langchain.com/python/integrations/) for your respective chat model.

### 提示缓存 (Prompt caching)

许多提供商提供提示缓存功能，以减少重复处理相同 token 时的延迟和成本。这些功能可以是**隐式**或**显式**的：

**详细说明：** 提示缓存是一种性能优化技术，其工作原理：

**工作机制**:

1. 提供商存储最近处理过的提示（或提示的一部分）
2. 当收到新请求时，检查是否有匹配的缓存内容
3. 如果命中缓存，跳过重新处理，直接使用缓存结果
4. 只处理缓存后的新内容

**优势**:

- **降低延迟**: 缓存命中时响应速度可提升 80-90%
- **减少成本**: 缓存的 token 通常按更低的价格计费（或免费）
- **提高吞吐量**: 服务器资源得到更有效利用

**适用场景**:

- 具有固定系统提示的对话应用
- 需要重复分析相同文档的场景
- 使用大量示例的 few-shot 学习
- RAG 系统（检索相同的上下文文档）

**注意事项**:

- 缓存通常有时间限制（如 5-15 分钟）
- 提示必须完全相同才能命中缓存
- 某些提供商要求最小 token 数量才启用缓存

- **Implicit prompt caching:** providers will automatically pass on cost savings if a request hits a cache. Examples: [OpenAI](/oss/python/integrations/chat/openai) and [Gemini](/oss/python/integrations/chat/google_generative_ai) (Gemini 2.5 and above).
- **Explicit caching:** providers allow you to manually indicate cache points for greater control or to guarantee cost savings. Examples: [`ChatOpenAI`](https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI/) (via `prompt_cache_key`), Anthropic's [`AnthropicPromptCachingMiddleware`](/oss/python/integrations/chat/anthropic#prompt-caching) and [`cache_control`](https://docs.langchain.com/oss/python/integrations/chat/anthropic#prompt-caching) options, [AWS Bedrock](/oss/python/integrations/chat/bedrock#prompt-caching), [Gemini](https://python.langchain.com/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html).

> **⚠️ 警告**: 提示缓存通常仅在超过最小输入 token 阈值时才启用。详见[提供商页面](/oss/python/integrations/chat)。

Cache usage will be reflected in the [usage metadata](/oss/python/langchain/messages#token-usage) of the model response.

### Token 使用量 (Token usage)

许多模型提供商将 token 使用信息作为调用响应的一部分返回。当可用时，此信息将包含在相应模型生成的 [`AIMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage) 对象中。有关更多详细信息，请参阅[消息指南](/oss/python/langchain/messages)。

**详细说明：** Token 使用量跟踪对于以下方面至关重要：

**为什么跟踪 Token 使用量**:

- **成本管理**: 大多数提供商按 token 计费，跟踪使用量可控制成本
- **性能优化**: 识别可以优化提示以减少 token 的地方
- **配额管理**: 监控是否接近速率限制
- **应用分析**: 了解不同功能的资源消耗

**Token 类型**:

- **输入 Token (Input tokens)**: 您发送给模型的文本（提示、消息历史、文档等）
- **输出 Token (Output tokens)**: 模型生成的文本
- **缓存 Token (Cached tokens)**: 从缓存中读取的 token（如果启用缓存）
- **推理 Token (Reasoning tokens)**: 某些模型在推理过程中使用的内部 token

**成本考虑**:

- 输出 token 通常比输入 token 更贵（可能是 2-3 倍）
- 缓存命中的 token 通常免费或大幅折扣
- 不同模型的 token 价格差异很大
- 批处理调用可能有折扣

> **📝 注意**: 某些提供商 API，特别是 OpenAI 和 Azure OpenAI 聊天补全，要求用户选择接收流式上下文中的 token 使用数据。详见集成指南的[流式使用元数据](/oss/python/integrations/chat/openai#streaming-usage-metadata)部分。

您可以使用回调或上下文管理器跟踪应用程序中跨模型的聚合 token 计数，如下所示：

**详细说明：** LangChain 提供了两种跟踪方式：

- **回调处理器**: 适合需要在整个应用生命周期中跟踪的场景
- **上下文管理器**: 适合跟踪特定代码块的使用量，自动清理资源

#### 使用回调处理器

```python
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

callback = UsageMetadataCallbackHandler()
result = model.invoke("你好", config={"callbacks": [callback]})
print(callback.usage_metadata)
```

输出示例:

```python
{
    'qwen-plus': {
        'input_tokens': 2,
        'output_tokens': 15,
        'total_tokens': 17
    }
}
```

#### 使用上下文管理器

```python
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback
import os
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

with get_usage_metadata_callback() as cb:
    model.invoke("你好")
    model.invoke("再见")
    print(cb.usage_metadata)
```

输出示例:

```python
{
    'qwen-plus': {
        'input_tokens': 4,
        'output_tokens': 30,
        'total_tokens': 34
    }
}
```
