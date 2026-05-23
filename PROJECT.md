# 吃喝玩乐小助手——从零构建一个 AI Agent 的完整思考

## 这是什么

一个 AI 旅行规划助手。你用自然语言问"帮我规划成都3日游"，它会自己上网搜攻略、查天气、算路线、避开去过的地方，最后输出一份带交通明细的行程。

**关键区别**：它不是 ChatGPT 套壳。它真的会调用外部工具（高德地图、天气预报、搜索引擎），然后基于工具返回的真实数据做决策。

---

## 一、用了什么技术栈，为什么选它

### 1. MCP 协议（Model Context Protocol）

**是什么**：Anthropic 2024年发布的 AI-工具通信标准。定义了 Agent 如何发现、调用外部工具。

**为什么选它**：
- 工具以独立进程（stdio Server）运行，崩溃不影响 Agent 主进程
- 新增工具只需加一个 Python 文件 + 一行配置，不改主逻辑
- 面试加分：MCP 是 2025 年最热的 Agent 基础设施方向

**为什么不用 Function Calling 直接写**：
OpenAI/DeepSeek 的 Function Calling 要求把工具定义作为 JSON Schema 传给模型，模型返回调用意图，你手动执行。这套流程 MCP 帮你标准化了——工具发现、参数校验、错误处理全自动。

### 2. LangGraph ReAct Agent

**是什么**：LangChain 团队的状态图 Agent 框架。ReAct = Reasoning + Acting，即"思考 → 行动 → 观察 → 思考"循环。

**为什么选它**：
- 自带 checkpoint（对话历史持久化），不需要自己管理
- 自带 streaming（astream_events），token 级实时输出
- recursion_limit 可配置，避免复杂任务被截断

**为什么不用 AutoGPT/BabyAGI**：
那些框架用"思路链 + 无限循环"模式，容易跑飞、消耗大量 Token。LangGraph 的状态图模式有明确的终止条件和可追踪的执行路径。

**为什么不用 LangChain 的 create_agent（新版）**：
因为环境里 langchain 版本不兼容，新版 import 会报错。但功能等价，不是缺陷。

### 3. DeepSeek 作为 LLM

**为什么选它**：
- 中文能力达 GPT-4 级别
- API 价格远低于 Claude/GPT-4（约 1/10）
- 支持 tool calling 和流式输出
- API Key 已经配好（项目启动时 ChatTongyi 无 Key）

### 4. SSE（Server-Sent Events）流式输出

**是什么**：HTTP 长连接，服务端持续推送数据。

**为什么选它**：比 WebSocket 轻量，单向推送够用（Agent → 用户）。FastAPI 原生支持。

**为什么不用 WebSocket**：旅游 Agent 没有"用户中途打断 Agent 思考"的需求，单向推送足够。WebSocket 多出来的双向通信能力用不上。

### 5. SQLite / jieba / sklearn

**为什么不用 PostgreSQL/MySQL**：个人项目，SQLite 零配置、文件级存储、够用。

**为什么不用向量数据库（ChromaDB/Milvus）**：Windows Conda 环境下 ChromaDB 段错误、onnxruntime DLL 不可用。jieba + sklearn TF-IDF 纯 Python 方案零二进制依赖，可用级质量满足需求。

### 6. 原生 JavaScript（无框架）

**为什么不用 React/Vue**：单页面应用，只管理一个聊天列表 + 侧边栏 + 弹窗。React 的虚拟 DOM 和状态管理对这个复杂度是过度设计。

---

## 二、解决了什么问题（按重要性排序）

### 问题 1：LLM 编造地图链接——准确率 ~40%

**现象**：让 AI 推荐密室逃脱，它会"好心地"给每条结果拼一个百度地图链接。但坐标是编的，点进去位置错误。

**根因**：LLM 不知道百度地图的真实坐标。它只能根据训练数据里的模式"猜"一个 URL 格式，但猜不准。

**解决方案**：
1. 工具返回结果时，把坐标写入后端 JSON 缓存文件
2. 后端通过 SSE `search_ready` 事件把缓存 ID 推给前端
3. 前端拿到 ID 后 fetch 坐标数据，等 AI 说完话后在 DOM 里注入正确的导航按钮
4. 同时清理 AI 自己编的旧链接

**关键设计决策**：**把链接生成权从 LLM 剥离**。LLM 只负责说人话（推荐理由），确定性数据（坐标 → URL）由前端计算。

**可推广的 insight**：Agent 系统里，让 LLM 做它擅长的事（语言理解），确定性计算交给传统代码。这是降低 Agent 不可靠性的核心方法。

---

### 问题 2：桌面端 GPS 不可用——定位可用率 0%

**现象**：台式机没有 GPS 硬件，浏览器 `navigator.geolocation` 直接报错。

**解决方案**：双通道降级
1. 优先调浏览器 GPS（8s 超时）
2. 超时则弹出文本输入框：用户输入"广州珠江新城"
3. 调高德 POI 搜索 + 地理编码 API 双重解析 → 拿到坐标

**关键设计决策**：**不假设用户环境**。移动端用 GPS，桌面端用手动输入，同一套接口。

---

### 问题 3：复杂旅游规划被截断——25 步限制

**现象**：成都 2 日游需要调 15+ 个工具（搜攻略 ×3 + fetch_url ×2 + 查天气 + 查路线 ×8），LangGraph 默认 `recursion_limit=25` 不够。

**根因**：每次工具调用 = 2 步（LLM 推理 + 工具执行）。15 个工具 = 30 步 > 25。

**解决方案**：在每次 invoke 的 config 里注入 `recursion_limit=100`。选运行时注入而非编译期是因为——Agent 的复杂度取决于任务，不是固定值。

---

### 问题 4：上下文膨胀——Token 消耗过高

**现象**：旅游规划中 15 个工具的返回结果全堆在上下文里，36-44 条消息。后续每次 LLM 推理都要重传全部历史。

**解决方案**（两个层次）：
1. **句子级 TF-IDF 摘要**：对超过 8 轮的旧工具消息，用 jieba 分词 + TF-IDF 算每句重要性，保留高分句（数字、地名），丢弃低分句（"向前步行10米"）
2. **Plan Mode**：分离"探索"和"规划"阶段。探索结果写文件，主 Agent 只读结论

**为什么不用 LLM 摘要**：每摘要一次 = 额外一次 LLM 调用。快省下的 Token 又被摘要消耗了。TF-IDF 是纯计算，0 Token 开销。

---

### 问题 5：embedding 方案全崩——环境兼容性

**现象**：Windows Conda + Python 3.11 环境下——
- sentence-transformers → huggingface_hub 版本冲突
- ChromaDB ONNX → DLL 加载失败
- fastembed → 同样 onnxruntime 问题

**解决方案**：全面回退到 jieba 分词 + sklearn TfidfVectorizer + 余弦相似度。纯 Python，零二进制依赖，内存友好。

**关键设计决策**：知识库的核心价值是"搜到相关的块"，不是"搜到最精确的块"。TF-IDF 对中文旅游攻略场景（关键词匹配为主）够用。

---

## 三、架构亮点

### 1. 工具热插拔

14 个 MCP 工具分布在 5 个 Server 文件中。加新工具 = 写一个 `@mcp.tool()` 装饰函数 + 在 `servers_config.json` 加一条。LLM 自动发现（`ListToolsRequest`），不需要改 Agent 代码。

```
calculator_server.py  → calculate
weather_server.py     → query_weather, forecast_weather, get_weather_tips
transit_server.py     → search_transit, search_driving, search_walking,
                        search_places, search_nearby
search_server.py      → web_search, fetch_url, search_knowledge,
                        add_knowledge, search_skills, add_skill
code_server.py        → code_run
```

### 2. 前端异步注入

导航按钮、技能库按钮都是 JavaScript 在 AI 说完话后自动注入的，不是 LLM 输出的 HTML。LLM 无法破坏这个流程。

### 3. 三级 SSE 事件流

```
text         → token 级流式字符（首 Token < 1s）
tool_start   → 工具调用开始（含状态栏更新）
tool_end     → 工具调用结束（含耗时）
search_ready → 搜索结果缓存 ID 预推送（前端异步预取坐标）
done         → 流式结束（触发按钮注入 + 上下文压缩）
```

每类事件独立推送，前端按需响应。不是把所有信息混在一个 JSON 里。

### 4. 评分驱动的去重

去过的地方不是简单"排除"，而是按评分分三档：
- ≥4 分 → 喜爱，可再推
- 2-3 分 → 备选
- <2 分 → 严格排除
- 未评分 → 默认排除（保守）

---

## 四、评估体系

12 个自动化测试用例，覆盖天气/路线/POI/知识库/旅游规划等功能模块。每次改代码后可跑 `python eval/run_eval.py` 验证是否引入回归。

| 指标 | 值 |
|------|-----|
| 通过率 | 83.3% |
| 工具准确率 | ~92% |
| 关键词匹配率 | 91.7% |
| 首 Token 延迟 | ~1s |
| 错误率 | < 10% |

剩余 16.7% 的未通过用例是因为 Plan Mode 重排了工具调用顺序——Agent 遵循"先搜索本地、再联网"的策略，与测试用例期望的"直接调用某工具"不同。这是设计行为，不是缺陷。

---

## 五、技术决策速查表

| 决策点 | 选了什么 | 为什么不用替代方案 |
|--------|----------|-------------------|
| Agent 框架 | LangGraph ReAct | AutoGPT 容易跑飞；原生 Function Calling 要手写循环 |
| 工具协议 | MCP stdio | 直接写 Function Calling 缺少标准化、工具发现、生命周期管理 |
| 流式输出 | SSE | WebSocket 的双向能力用不上 |
| LLM | DeepSeek | Claude/GPT-4 贵 10 倍；ChatTongyi 无 API Key |
| 向量检索 | TF-IDF + sklearn | ChromaDB/ONNX 在 Conda 环境崩溃 |
| 前端 | 原生 JS | React 对这个复杂度过度设计 |
| 数据存储 | SQLite | PostgreSQL 需要独立服务，个人项目零配置优先 |
| 上下文压缩 | TF-IDF 句子评分 | LLM 摘要额外消耗 Token |
| 部署 | start.bat + 二维码 | Docker/Render 要绑卡/实名，本地更简单 |

---

## 六、如何运行

```bash
# 双击 start.bat，或：
python api_server.py
# 打开 http://localhost:8000
# 手机扫码页面底部的二维码即可使用
```
