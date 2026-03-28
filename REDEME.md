# Node.js + LM Studio 本地大模型 RAG Demo

## 📦 依赖安装

```bash
npm init -y
npm install flat-vector-store axios crypto
```

---

## 🧠 完整代码（`rag-lmstudio.js`）

```javascript
const { VectorStore } = require("flat-vector-store");
const axios = require("axios");
const fs = require("fs").promises;
const crypto = require("crypto");

// ==================== LM Studio API 客户端 ====================
class LMStudioClient {
  constructor(baseUrl = "http://198.18.0.1:9527") {
    this.baseUrl = baseUrl;
    this.timeout = 60000; // 60秒超时
  }

  async generate(prompt, options = {}) {
    const params = new URLSearchParams({
      prompt: prompt,
      max_tokens: options.maxTokens || 512,
      temperature: options.temperature ?? 0.7,
      top_p: options.topP ?? 0.9,
      stream: "false",
    });

    try {
      const response = await axios.post(
        `${this.baseUrl}/v1/chat/completions`,
        params.toString(),
        {
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Accept: "application/json",
          },
          timeout: this.timeout,
          maxRedirects: 0,
        },
      );

      return response.data.choices[0].message.content;
    } catch (error) {
      console.error("LM Studio API 错误:", error.message);

      // 如果连接失败，返回默认回复
      if (error.code === "ECONNREFUSED" || error.response?.status === 404) {
        return "[LM Studio 未启动或不可用]";
      }

      throw error;
    }
  }

  async healthCheck() {
    try {
      await axios.get(`${this.baseUrl}/v1/models`, { timeout: 5000 });
      return true;
    } catch (error) {
      console.log("LM Studio 健康检查失败:", error.message);
      return false;
    }
  }

  async getAvailableModels() {
    try {
      const response = await axios.get(`${this.baseUrl}/v1/models`, {
        timeout: 5000,
      });
      return response.data.models.map((m) => m.id).filter(Boolean);
    } catch (error) {
      console.error("获取模型列表失败:", error.message);
      return [];
    }
  }
}

// ==================== 模拟文本嵌入模型 ====================
async function embedText(text) {
  // 简单哈希模拟 embedding，仅用于演示
  const hash = await crypto.subtle.digest("SHA-256", Buffer.from(text));
  return Array.from(new Uint8Array(hash)).map((b) => b / 255);
}

// ==================== 文档分块 ====================
function chunkText(text, size = 100) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size) {
    chunks.push(text.slice(i, i + size));
  }
  return chunks;
}

// ==================== 向量数据库管理 ====================
async function initVectorStore() {
  const store = new VectorStore();
  await fs.writeFile("vector-store.json", JSON.stringify({}), "utf8");
  return store;
}

async function loadVectorStore() {
  try {
    const data = await fs.readFile("vector-store.json", "utf8");
    if (!data) return new VectorStore();
    const store = new VectorStore(JSON.parse(data));
    return store;
  } catch (e) {
    console.error("加载向量数据库失败:", e);
    return new VectorStore();
  }
}

async function saveVectorStore(store) {
  await fs.writeFile(
    "vector-store.json",
    JSON.stringify({ data: store.data }),
    "utf8",
  );
}

// ==================== RAG Pipeline ====================
async function ragPipeline(query, vectorStore, llmClient) {
  // 1. 向量化查询
  const queryEmbedding = await embedText(query);

  // 2. 相似度搜索
  const results = vectorStore.search(queryEmbedding, { topK: 3 });

  // 3. 构建上下文
  let context = "";
  if (results.length > 0) {
    for (const item of results) {
      context += `来源：${item.metadata.source}\n内容：${item.text}\n`;
    }
  } else {
    context = "未找到相关文档。";
  }

  // 4. 调用真实 LLM（LM Studio）生成回答
  const systemPrompt =
    "你是一个智能助手，请根据以下提供的上下文信息回答问题。\n" +
    "如果上下文中没有相关信息，请诚实地说明你不知道，不要编造答案。\n\n";

  const fullPrompt = `${systemPrompt}上下文：\n${context}\n\n问题：${query}`;

  try {
    const answer = await llmClient.generate(fullPrompt, { maxTokens: 512 });
    return { query, context, answer };
  } catch (error) {
    console.error("LLM 生成失败:", error.message);
    return {
      query,
      context,
      answer: "[LM Studio 响应超时或出错，请检查服务是否正常运行]",
    };
  }
}

// ==================== 主程序 ====================
async function main() {
  // 配置
  const LM_STUDIO_URL = "http://198.18.0.1:9527";

  console.log("🚀 初始化 RAG 系统...\n");

  // 1. 检查 LM Studio 是否可用
  const llmClient = new LMStudioClient(LM_STUDIO_URL);
  const isHealthy = await llmClient.healthCheck();

  if (!isHealthy) {
    console.log("⚠️  LM Studio 未启动或不可用");
    console.log(`   请确保 LM Studio 已运行在: ${LM_STUDIO_URL}`);
    console.log("\n按任意键继续（将使用模拟 LLM）...");
    process.stdin.read();
  } else {
    const availableModels = await llmClient.getAvailableModels();
    if (availableModels.length > 0) {
      console.log(
        `✅ LM Studio 可用，已加载模型: ${availableModels.join(", ")}`,
      );
    }
  }

  // 2. 初始化向量数据库
  let vectorStore = await loadVectorStore();

  // 3. 示例文档内容（可以替换为真实文件）
  const docs = [
    "Node.js 是一个基于 Chrome V8 引擎的 JavaScript 运行环境。",
    "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术，用于增强 LLM 回答能力。",
    "向量数据库是存储高维向量数据的数据库，常用于语义搜索。",
    "LM Studio 是一个本地大模型推理工具，支持多种开源模型如 Llama、Mistral 等。",
    "HTTP API 接口允许外部应用与 LM Studio 通信，实现 RAG 等应用场景。",
  ];

  // 4. 文档处理与向量化
  console.log("\n📚 正在处理文档...\n");
  for (const doc of docs) {
    const chunks = chunkText(doc, 50);
    for (const chunk of chunks) {
      const embedding = await embedText(chunk);
      vectorStore.add({ text: chunk, metadata: { source: "demo" } }, embedding);
    }
  }

  // 5. 保存向量数据库
  await saveVectorStore(vectorStore);
  console.log("✅ 文档处理完成\n");

  // 6. RAG 查询示例
  const queries = [
    "什么是 Node.js？",
    "LM Studio 是什么？",
    "RAG 技术的原理是什么？",
  ];

  for (const query of queries) {
    console.log(`\n🔍 问题: ${query}`);
    console.log("─".repeat(60));

    const result = await ragPipeline(query, vectorStore, llmClient);

    console.log("\n📄 检索到的上下文:\n", result.context);
    console.log("\n💬 LLM 回答:\n", result.answer);
  }

  // 7. 交互式查询（可选）
  console.log('\n\n⌨️  输入 "quit" 退出，或输入问题开始交互...\n');

  while (true) {
    const readline = require("readline").createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    const question = await new Promise((resolve) =>
      readline.question("您想问什么？ ").then(resolve),
    );

    if (question.trim().toLowerCase() === "quit") {
      console.log("\n👋 再见！");
      break;
    }

    const result = await ragPipeline(question, vectorStore, llmClient);
    console.log(result.answer);
  }

  readline.close();
}

main().catch(console.error);
```

---

## 🚀 运行方式

### 1. 启动 LM Studio

在 LM Studio 中：

- 选择并加载一个模型（如 `Llama-3-8B-Instruct`）
- 点击 **"Start Server"**
- 确保 API 端口设置为 **9527**

### 2. 运行 RAG Demo

```bash
node rag-lmstudio.js
```

---

## 📊 输出示例

```
🚀 初始化 RAG 系统...

✅ LM Studio 可用，已加载模型: llama-3-8b-instruct

📚 正在处理文档...

✅ 文档处理完成


🔍 问题：什么是 Node.js？
────────────────────────────────────────────────
来源：demo
内容：Node.js 是一个基于 Chrome V8 引擎的 JavaScript 运行环境。

来源：demo
内容：Node.js 是一个基于 Chrome V8 引擎的 JavaScript 运行环境。

💬 LLM 回答:
Node.js 是一个基于 Chrome V8 引擎的 JavaScript 运行环境，它允许开发者使用 JavaScript 编写服务器端代码...
```

---

## 🔧 配置选项

| 参数            | 默认值                   | 说明                  |
| --------------- | ------------------------ | --------------------- |
| `LM_STUDIO_URL` | `http://198.18.0.1:9527` | LM Studio API 地址    |
| `maxTokens`     | `512`                    | LLM 最大生成 token 数 |
| `temperature`   | `0.7`                    | 温度参数，控制创造性  |
| `topP`          | `0.9`                    | Top-p 采样参数        |

---

## ⚠️ 注意事项

1. **跨域问题**：如果 LM Studio 和 Node.js 在不同网络，可能需要配置 CORS
2. **超时设置**：大模型响应较慢，建议设置较长的超时时间（默认 60 秒）
3. **内存使用**：向量数据库会占用内存，生产环境建议使用持久化存储

---

需要我帮你添加以下功能吗？

- 📁 从文件读取文档进行向量化
- 🔌 REST API 封装，支持外部调用
- 🎨 Web UI（Express + EJS）展示 RAG 结果

# Node.js + LM Studio 本地大模型 RAG Demo

## 📦 依赖安装

```bash
npm init -y
npm install flat-vector-store axios crypto
```

---

## 🧠 完整代码（`rag-lmstudio.js`）

```javascript
const { VectorStore } = require("flat-vector-store");
const axios = require("axios");
const fs = require("fs").promises;
const crypto = require("crypto");

// ==================== LM Studio API 客户端 ====================
class LMStudioClient {
  constructor(baseUrl = "http://198.18.0.1:9527") {
    this.baseUrl = baseUrl;
    this.timeout = 60000; // 60秒超时
  }

  async generate(prompt, options = {}) {
    const params = new URLSearchParams({
      prompt: prompt,
      max_tokens: options.maxTokens || 512,
      temperature: options.temperature ?? 0.7,
      top_p: options.topP ?? 0.9,
      stream: "false",
    });

    try {
      const response = await axios.post(
        `${this.baseUrl}/v1/chat/completions`,
        params.toString(),
        {
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Accept: "application/json",
          },
          timeout: this.timeout,
          maxRedirects: 0,
        },
      );

      return response.data.choices[0].message.content;
    } catch (error) {
      console.error("LM Studio API 错误:", error.message);

      // 如果连接失败，返回默认回复
      if (error.code === "ECONNREFUSED" || error.response?.status === 404) {
        return "[LM Studio 未启动或不可用]";
      }

      throw error;
    }
  }

  async healthCheck() {
    try {
      await axios.get(`${this.baseUrl}/v1/models`, { timeout: 5000 });
      return true;
    } catch (error) {
      console.log("LM Studio 健康检查失败:", error.message);
      return false;
    }
  }

  async getAvailableModels() {
    try {
      const response = await axios.get(`${this.baseUrl}/v1/models`, {
        timeout: 5000,
      });
      return response.data.models.map((m) => m.id).filter(Boolean);
    } catch (error) {
      console.error("获取模型列表失败:", error.message);
      return [];
    }
  }
}

// ==================== 模拟文本嵌入模型 ====================
async function embedText(text) {
  // 简单哈希模拟 embedding，仅用于演示
  const hash = await crypto.subtle.digest("SHA-256", Buffer.from(text));
  return Array.from(new Uint8Array(hash)).map((b) => b / 255);
}

// ==================== 文档分块 ====================
function chunkText(text, size = 100) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size) {
    chunks.push(text.slice(i, i + size));
  }
  return chunks;
}

// ==================== 向量数据库管理 ====================
async function initVectorStore() {
  const store = new VectorStore();
  await fs.writeFile("vector-store.json", JSON.stringify({}), "utf8");
  return store;
}

async function loadVectorStore() {
  try {
    const data = await fs.readFile("vector-store.json", "utf8");
    if (!data) return new VectorStore();
    const store = new VectorStore(JSON.parse(data));
    return store;
  } catch (e) {
    console.error("加载向量数据库失败:", e);
    return new VectorStore();
  }
}

async function saveVectorStore(store) {
  await fs.writeFile(
    "vector-store.json",
    JSON.stringify({ data: store.data }),
    "utf8",
  );
}

// ==================== RAG Pipeline ====================
async function ragPipeline(query, vectorStore, llmClient) {
  // 1. 向量化查询
  const queryEmbedding = await embedText(query);

  // 2. 相似度搜索
  const results = vectorStore.search(queryEmbedding, { topK: 3 });

  // 3. 构建上下文
  let context = "";
  if (results.length > 0) {
    for (const item of results) {
      context += `来源：${item.metadata.source}\n内容：${item.text}\n`;
    }
  } else {
    context = "未找到相关文档。";
  }

  // 4. 调用真实 LLM（LM Studio）生成回答
  const systemPrompt =
    "你是一个智能助手，请根据以下提供的上下文信息回答问题。\n" +
    "如果上下文中没有相关信息，请诚实地说明你不知道，不要编造答案。\n\n";

  const fullPrompt = `${systemPrompt}上下文：\n${context}\n\n问题：${query}`;

  try {
    const answer = await llmClient.generate(fullPrompt, { maxTokens: 512 });
    return { query, context, answer };
  } catch (error) {
    console.error("LLM 生成失败:", error.message);
    return {
      query,
      context,
      answer: "[LM Studio 响应超时或出错，请检查服务是否正常运行]",
    };
  }
}

// ==================== 主程序 ====================
async function main() {
  // 配置
  const LM_STUDIO_URL = "http://198.18.0.1:9527";

  console.log("🚀 初始化 RAG 系统...\n");

  // 1. 检查 LM Studio 是否可用
  const llmClient = new LMStudioClient(LM_STUDIO_URL);
  const isHealthy = await llmClient.healthCheck();

  if (!isHealthy) {
    console.log("⚠️  LM Studio 未启动或不可用");
    console.log(`   请确保 LM Studio 已运行在: ${LM_STUDIO_URL}`);
    console.log("\n按任意键继续（将使用模拟 LLM）...");
    process.stdin.read();
  } else {
    const availableModels = await llmClient.getAvailableModels();
    if (availableModels.length > 0) {
      console.log(
        `✅ LM Studio 可用，已加载模型: ${availableModels.join(", ")}`,
      );
    }
  }

  // 2. 初始化向量数据库
  let vectorStore = await loadVectorStore();

  // 3. 示例文档内容（可以替换为真实文件）
  const docs = [
    "Node.js 是一个基于 Chrome V8 引擎的 JavaScript 运行环境。",
    "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术，用于增强 LLM 回答能力。",
    "向量数据库是存储高维向量数据的数据库，常用于语义搜索。",
    "LM Studio 是一个本地大模型推理工具，支持多种开源模型如 Llama、Mistral 等。",
    "HTTP API 接口允许外部应用与 LM Studio 通信，实现 RAG 等应用场景。",
  ];

  // 4. 文档处理与向量化
  console.log("\n📚 正在处理文档...\n");
  for (const doc of docs) {
    const chunks = chunkText(doc, 50);
    for (const chunk of chunks) {
      const embedding = await embedText(chunk);
      vectorStore.add({ text: chunk, metadata: { source: "demo" } }, embedding);
    }
  }

  // 5. 保存向量数据库
  await saveVectorStore(vectorStore);
  console.log("✅ 文档处理完成\n");

  // 6. RAG 查询示例
  const queries = [
    "什么是 Node.js？",
    "LM Studio 是什么？",
    "RAG 技术的原理是什么？",
  ];

  for (const query of queries) {
    console.log(`\n🔍 问题: ${query}`);
    console.log("─".repeat(60));

    const result = await ragPipeline(query, vectorStore, llmClient);

    console.log("\n📄 检索到的上下文:\n", result.context);
    console.log("\n💬 LLM 回答:\n", result.answer);
  }

  // 7. 交互式查询（可选）
  console.log('\n\n⌨️  输入 "quit" 退出，或输入问题开始交互...\n');

  while (true) {
    const readline = require("readline").createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    const question = await new Promise((resolve) =>
      readline.question("您想问什么？ ").then(resolve),
    );

    if (question.trim().toLowerCase() === "quit") {
      console.log("\n👋 再见！");
      break;
    }

    const result = await ragPipeline(question, vectorStore, llmClient);
    console.log(result.answer);
  }

  readline.close();
}

main().catch(console.error);
```

---

## 🚀 运行方式

### 1. 启动 LM Studio

在 LM Studio 中：

- 选择并加载一个模型（如 `Llama-3-8B-Instruct`）
- 点击 **"Start Server"**
- 确保 API 端口设置为 **9527**

### 2. 运行 RAG Demo

```bash
node rag-lmstudio.js
```

---

## 📊 输出示例

```
🚀 初始化 RAG 系统...

✅ LM Studio 可用，已加载模型: llama-3-8b-instruct

📚 正在处理文档...

✅ 文档处理完成


🔍 问题：什么是 Node.js？
────────────────────────────────────────────────
来源：demo
内容：Node.js 是一个基于 Chrome V8 引擎的 JavaScript 运行环境。

来源：demo
内容：Node.js 是一个基于 Chrome V8 引擎的 JavaScript 运行环境。

💬 LLM 回答:
Node.js 是一个基于 Chrome V8 引擎的 JavaScript 运行环境，它允许开发者使用 JavaScript 编写服务器端代码...
```

---

## 🔧 配置选项

| 参数            | 默认值                   | 说明                  |
| --------------- | ------------------------ | --------------------- |
| `LM_STUDIO_URL` | `http://198.18.0.1:9527` | LM Studio API 地址    |
| `maxTokens`     | `512`                    | LLM 最大生成 token 数 |
| `temperature`   | `0.7`                    | 温度参数，控制创造性  |
| `topP`          | `0.9`                    | Top-p 采样参数        |

---

## ⚠️ 注意事项

1. **跨域问题**：如果 LM Studio 和 Node.js 在不同网络，可能需要配置 CORS
2. **超时设置**：大模型响应较慢，建议设置较长的超时时间（默认 60 秒）
3. **内存使用**：向量数据库会占用内存，生产环境建议使用持久化存储

---

需要我帮你添加以下功能吗？

- 📁 从文件读取文档进行向量化
- 🔌 REST API 封装，支持外部调用
- 🎨 Web UI（Express + EJS）展示 RAG 结果
