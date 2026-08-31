# 验证探针

一次性的 API/性能验证脚本，不属于生产代码。

- `probe-minimal.mjs` —— 步骤 1：真实 DeepSeek API + 自定义 Python tool 的最小闭环

跑法（需 `.env` 里的 `DEEPSEEK_API_KEY`）：
```bash
set -a && . ./.env && set +a && node probes/probe-minimal.mjs
```
