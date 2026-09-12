# Lite Me / Eazo

体验入口：https://lite-me.vercel.app

Python FastAPI + HTML/CSS/JavaScript。线上使用 Vercel 和 Neon PostgreSQL，
账号、会话、个人资料、历史记录、压缩后的饮食图片和最新 Agent 轨迹均持久保存。
知识库是随代码部署的公开语料和向量，不依赖个人数据库中的预置记录。

## 本地运行

使用 Python 3.12，安装 `requirements.txt`，复制 `.env.example` 为 `.env` 并填写模型服务配置。

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn backend.main:app --reload
```

本地没有 `DATABASE_URL` 时使用 SQLite。线上必须提供 Neon 注入的 `DATABASE_URL`，
不会回退到临时文件数据库。`DEMO_MODE=false` 使用真实北京时间；演示时才设为 `true`。

## Vercel 部署

项目 `lite-me`，区域 `sin1`。环境变量设置在 Vercel 后端，不能写入前端或提交到 Git：

- `DATABASE_URL`：Neon 集成自动注入。
- `LLM_API_KEY`、`LLM_BASE_URL`：模型服务。
- `CHAT_MODEL`、`VISION_MODEL`、`EMBED_MODEL`：分别支持工具调用、图片识别、向量检索。
- `DEMO_MODE=false`、`APP_TIMEZONE=Asia/Shanghai`。

```sh
npx vercel link
npx vercel deploy --prod
```

`.vercelignore` 仅包含运行代码、前端、公开知识库及依赖配置。
本地密钥、账号数据库、上传图片、日志、路演材料和过程文件不进入部署包。
数据库免费档有用量限制，模型调用费用独立计算。

线上当前通过百炼北京标准接口 `https://dashscope.aliyuncs.com/compatible-mode/v1`
调用模型。聊天使用 `qwen-plus-2025-07-28`，图片使用 `qwen-vl-max`，
向量使用 `text-embedding-v4`。这些模型的额度与有效期在百炼控制台单独管理。

依赖由 `uv.lock` 固定；修改 `pyproject.toml` 后运行 `uv lock --python 3.12`，
并提交更新后的锁文件。Vercel 构建会按锁文件安装依赖。

## 验证

```sh
.venv/bin/pip install httpx
.venv/bin/python -m unittest scripts.test_deployment scripts.test_safety
```

部署后检查 `/api/health`，并验证注册、资料保存、重新登录、历史记录和真实模型生成。
模型失败时线上返回可重试错误，不将通用兜底内容存成个性化建议。
知识库检索依赖向量模型与 `data/corpus.npz` 一致；换向量模型时需重新构建公开语料向量。
