const fs = require("fs");

// ==================== LM Studio API 客户端 ====================
class LMStudioClient {
  constructor(baseUrl = "http://198.18.0.1:9527") {
    this.baseUrl = baseUrl;
    this.timeout = 60000; // 60 秒超时
  }

  async generate(prompt, options = {}) {
    const payload = {
      messages: [{ role: "user", content: prompt }],
      max_tokens: options.maxTokens || 512,
      temperature: options.temperature ?? 0.7,
      top_p: options.topP ?? 0.9,
    };

    try {
      const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(payload),
        redirect: "manual",
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
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
      const response = await fetch(`${this.baseUrl}/v1/models`, {
        method: "GET",
        redirect: "manual",
      });
      return response.ok;
    } catch (error) {
      console.log("LM Studio 健康检查失败:", error.message);
      return false;
    }
  }

  async getAvailableModels() {
    try {
      const response = await fetch(`${this.baseUrl}/v1/models`, {
        method: "GET",
        redirect: "manual",
      });
      if (!response.ok) return [];
      const data = await response.json();
      return data.models?.map((m) => m.id).filter(Boolean) || [];
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

// ==================== 向量数据库管理（纯原生实现）====================
class SimpleVectorStore {
  constructor(data = null) {
    this.data = data || [];
  }

  add(item, embedding) {
    this.data.push({ text: item.text, metadata: item.metadata, embedding });
  }

  search(embedding, options = {}) {
    const topK = options.topK || 3;

    // 计算相似度（简化版：使用点积）
    let results = this.data.map((item) => {
      const similarity = embedding.reduce(
        (a, b) => a + b * item.embedding[0],
        0,
      );
      return { ...item, score: similarity };
    });

    // 按相似度降序排序并取前 K 个
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK).map(({ score, ...rest }) => rest);
  }
}

async function initVectorStore() {
  const store = new SimpleVectorStore();
  await fs.promises.writeFile("vector-store.json", JSON.stringify({}), "utf8");
  return store;
}

async function loadVectorStore() {
  try {
    const data = await fs.promises.readFile("vector-store.json", "utf8");
    if (!data) return new SimpleVectorStore();
    const parsed = JSON.parse(data);
    return new SimpleVectorStore(parsed.data || []);
  } catch (e) {
    console.error("加载向量数据库失败:", e.message);
    return new SimpleVectorStore();
  }
}

async function saveVectorStore(store) {
  await fs.promises.writeFile(
    "vector-store.json",
    JSON.stringify({ data: store.data }),
    "utf8"
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
    "你是一个智能助手，请根据以下提供的上下文信息回答问题。\n"
    + "如果上下文中没有相关信息，请诚实地说明你不知道，不要编造答案。\n\n";

  const fullPrompt = `${systemPrompt}上下文：\n${context}\n\n问题：${query}`;

  try {
    const response = await llmClient.generate(fullPrompt, { maxTokens: 512 });
    // 提取回答内容
    let answerText = "";
    if (response.choices && response.choices[0] && response.choices[0].message) {
      answerText = response.choices[0].message.content || "";
    } else if (typeof response === 'string') {
      answerText = response;
    }
    return { query, context, answer: answerText };
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
    console.log(`   请确保 LM Studio 已运行在：${LM_STUDIO_URL}`);
    console.log("\n按任意键继续（将使用模拟 LLM）...");
    process.stdin.read();
  } else {
    const availableModels = await llmClient.getAvailableModels();
    if (availableModels.length > 0) {
      console.log(
        `✅ LM Studio 可用，已加载模型：${availableModels.join(", ")}`,
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
    console.log(`\n🔍 问题：${query}`);
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

    // readline.question 可能返回 Promise 或直接字符串，需要兼容处理
    let question;
    if (typeof readline.question === 'function') {
      try {
        const result = await new Promise((resolve) => {
          readline.question("您想问什么？ ", resolve);
        });
        question = result || "";
      } catch (e) {
        console.log("交互模式出错，将跳过...");
        break;
      }
    } else {
      // 降级：直接读取一行
      process.stdin.setEncoding('utf8');
      question = await new Promise((resolve) => {
        const chunk = [];
        let running = true;
        while (running) {
          const data = process.stdin.read();
          if (!data) break;
          chunk.push(data);
          // 检测换行符
          if (chunk.join('').includes('\n')) {
            question = chunk.join('').slice(0, -1);
            running = false;
          }
        }
      });
    }

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
