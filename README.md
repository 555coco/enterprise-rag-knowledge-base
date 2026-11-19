# Enterprise RAG Knowledge Base

一个面向面试展示的、可运行且可量化的 RAG 知识库基础版本。内置中文/英文关键词检索、来源引用、黄金数据集评估、租户过滤、幂等入库与请求指标。默认回答器不需要 API key；配置后可以调用兼容 OpenAI Chat Completions 的模型。

**边界：** 当前是本地 MVP，尚未实现可信身份认证、文档 ACL、向量检索、异步摄取、分布式追踪或生产压测。详见 [架构与验收](docs/architecture.md)。

## 立即运行离线演示

Python 3.11+ 与内置 SQLite FTS5 即可，无需安装依赖：

```bash
PYTHONPATH=src python -m enterprise_rag.demo
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 启动 API

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
export RAG_API_KEY=local-demo-key
uvicorn enterprise_rag.api:app --reload
```

打开 `http://127.0.0.1:8000/docs` 查看接口，或者：

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H 'Content-Type: application/json' -H 'X-API-Key: local-demo-key' -H 'X-Tenant-Id: demo' \
  -d '{"title":"退货政策","content":"签收后的30天内可以申请退货。"}'

curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' -H 'X-API-Key: local-demo-key' -H 'X-Tenant-Id: demo' \
  -d '{"question":"退货期限是多久？"}'
```

`POST /evaluate` 接收 `{"cases":[{"question":"...","expected_document_id":"文档 ID","answer_contains":["关键词"]}],"k":5}`；`GET /metrics` 返回请求量、错误数与延迟统计。当前评估只测检索和可检查的答案特征，不声称能自动测得事实正确性。

## 接入模型

按 [.env.example](.env.example) 设置 `RAG_ANSWER_PROVIDER=openai_compatible`、`RAG_LLM_BASE_URL`、`RAG_LLM_API_KEY`、`RAG_LLM_MODEL`。只发送检索到的证据给模型；模型结果必须包含有效的 `[1]` 等引用，失败返回 503 并记录错误类型。

## 代码结构

- `src/enterprise_rag/core.py`：摄取、索引、检索、回答基线、评估和指标。
- `src/enterprise_rag/providers.py`：可替换的模型回答器。
- `src/enterprise_rag/api.py`：FastAPI 接口与请求校验。
- `datasets/`：演示文档和评估案例。
- `tests/`：核心行为测试。

本项目不含真实企业数据、模型密钥或个人信息。上传公开仓库前请只提交 `.env.example`，不要提交 `.env` 或数据库文件。
