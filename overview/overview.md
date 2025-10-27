# LangChain 生态概述（v1.0版本）

LangChain 是一个用于开发由大型语言模型（LLMs）驱动的应用程序的框架。LangChain 1.0 是一个**聚焦 Agent 开发、生产就绪**的重大版本，整个生态包含三个核心产品：

- **LangChain**：高层抽象框架，提供预构建的 Agent 架构和模型集成，用不到 10 行代码即可快速构建 Agent 应用
- **LangGraph**：低层编排框架和运行时，用于构建、管理和部署长期运行的有状态 Agent，支持持久化执行、流式处理和人机协作
- **LangSmith**：开发者平台，提供调试、监控和评估功能，帮助开发者优化和部署应用

![Langchain framework](images/framework.png)

## LangChain 1.0 的架构特点

LangChain 1.0 采用了**分层架构设计**，Agent 构建在 LangGraph 之上，这种设计让开发者可以：
使用 LangChain 的高级抽象快速原型开发，需要时"下沉"到 LangGraph 进行深度定制，通过 LangSmith 获得完整的调试和监控能力

## LangChain 1.0 核心组件

### 1. **Models（模型）- 统一的模型接口**

支持多种语言模型，如 OpenAI、Anthropic、Google、Mistral、Llama 等。LangChain 标准化了不同提供商的 API，让你可以：

- 无缝切换模型提供商，避免供应商锁定
- 使用统一的接口访问不同模型的能力
- 通过 `content_blocks` 属性统一访问现代 LLM 特性（如多模态输入、结构化输出等）

### 2. **Agents（代理）- 全新的标准方式** 🆕

LangChain 1.0 引入的核心改进，替代了之前的 `create_react_agent`：

- 用不到 10 行代码即可构建 Agent
- 提供清晰、强大的 API 接口
- 基于 LangGraph 构建，自动获得持久化执行、流式处理、人机协作等能力
- 同时保持足够的灵活性支持复杂的上下文工程

### 3. **Tools（工具）**

提供访问外部资源的能力，扩展模型的功能：

- API 调用、Google 搜索、SQL 数据库查询等
- 支持自定义工具开发
- Agent 可以动态选择合适的工具来完成任务

### 4. **Memory（记忆）与 State（状态管理）**

通过 LangGraph 的状态管理能力实现：

- **短期记忆**：当前会话的上下文信息
- **长期记忆**：结合向量数据库持久化存储重要信息
- **Checkpointing**：支持会话恢复和状态持久化
- **Human-in-the-Loop**：支持人工干预和审核

### 5. **Prompt Templates（提示词模板）**

创建动态提示词，提高模型的泛化能力：

- 模板化方式根据输入生成相应的提示词
- 引导模型生成更准确的输出
- 支持复杂的提示词工程

### 6. **Chains（链式调用）与 LCEL**

LangChain Expression Language（LCEL）提供声明式的链式编排：

- 支持批处理、并行化和自动重试
- 原生支持流式处理（token 级别和中间步骤）
- 高度可组合，便于构建复杂的处理流程

### 7. 

## LangChain 1.0 的核心架构

LangChain 1.0 重构了整体架构，采用**模块化、分层设计**，核心架构包含以下关键模块：

### 📦 包结构（简化后）

#### 1. **langchain-core** - 核心基础

提供稳定的核心抽象和接口：

- 模型接口的标准化抽象
- 工具（Tools）、向量存储（Vector Stores）等核心组件
- LangChain Expression Language（LCEL）基础
- 轻量级设计，便于扩展和集成

#### 2. **langchain** - 主包（简化版）

LangChain 1.0 简化了主包，聚焦于 Agent 开发的核心功能：

- `create_agent` - 新的标准 Agent 构建方式
- 预构建的 Agent 架构
- 基于 LangGraph 的 Agent 实现
- 核心链式调用组件

#### 3. **langchain-classic** - 历史遗留功能 🆕

将 0.x 版本的历史功能迁移到独立包：

- 保持向后兼容性
- 便于逐步迁移到新架构
- 不影响新项目使用最新特性

#### 4. **langchain-community** - 社区集成

整合社区贡献的第三方工具和集成：

- 各种模型提供商的集成
- 向量数据库集成
- 文件解析器、文本分割器等工具
- 丰富的生态系统支持

### 生态系统架构

#### **LangGraph** - Agent 编排运行时

LangChain 1.0 的 Agent 构建在 LangGraph 之上：

- 图结构的 Agent 编排
- 持久化执行（Durable Execution）
- 状态管理和检查点机制
- 流式处理和人机协作支持
- 适用于复杂、长期运行的 Agent 应用

#### **LangSmith** - 可观测性平台

完整的开发者工具链：

- **调试**：可视化执行路径，捕获状态转换
- **监控**：生产环境的性能监控和告警
- **评估**：Agent 轨迹评估和性能分析
- **优化**：详细的运行时指标和 Token 使用统计

#### **LangServe** - 部署服务

将 LangChain 应用部署为生产服务：

- 基于 FastAPI 的 REST API
- 支持流式传输和批量处理
- 易于集成到现有系统
- 生产级的性能和稳定性


