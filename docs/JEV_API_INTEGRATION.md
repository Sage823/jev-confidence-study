# Jev API 接入说明

## 官方接口

当前适配器遵循 TypeSafe 官方 [Quick start](https://docs.typesafe.ai/introduction/quickstart)：

- `POST https://api.typesafe.ai/v1/systemone`
- Header: `Authorization: Bearer <API_KEY>` 和 `Content-Type: application/json`
- 请求包含 `model`、`state`、`questions`。
- `Choice` 问题通过 `criteria` 提供选项键与选项文本；响应给出 `choice`、选项概率及 `usage`。

本实现直接用 Python 标准库 `urllib`，不需要 SDK 或运行时依赖。每次调用都必须显式运行 `run-jev`；导入 API adapter 或预览 payload 不会发网络请求。实现没有自动重试，避免超时后不确定是否已计费时重复提交。

## 输入文件

`--input` 接收 UTF-8 JSONL，每行一条规范化题目记录。最少需要：

```json
{"item_id":"sciq-test-1","domain":"science","split":"test","question":"Which planet is known as the Red Planet?","options":{"A":"Venus","B":"Mars","C":"Jupiter","D":"Mercury"},"gold_key":"B","option_order":["A","B","C","D"],"source":{"dataset":"SciQ","revision":"..."}}
```

`gold_key` 只用于本地评价，不会放进 API 请求。`options` 的键必须稳定；`option_order` 记录展示顺序。不同模型应使用同一份 item 文件和同一顺序条件。

## 预览请求（零 API 调用）

```powershell
$env:PYTHONPATH = "src"
python -m jevstudy jev-payload path\to\one-item.json --model jev-1.13.0
```

`jev-payload` 当前接受单个 JSON item 文件，而非 JSONL。检查请求中题目和选项无误后再调用 API。模型版本必须显式填写，便于复现实验；不要用会漂移的 `latest` alias 作为正式实验的默认选择。

## 发起真实调用（会消耗 API）

先在当前 PowerShell 会话设置 key；不要放在代码文件、输入 JSONL 或提交历史中：

```powershell
$env:TYPESAFE_API_KEY = "从 TypeSafe 控制台取得的 key"
$env:PYTHONPATH = "src"
python -m jevstudy run-jev --input data\items.jsonl --out runs\jev-pilot --model jev-1.13.0 --limit 2
```

`--limit 2` 适合第一轮真实 smoke run。确认 `predictions.jsonl` 与 `errors.jsonl` 后，再移除限制运行完整冻结文件。输出必须写入 `runs/` 下的新目录，程序拒绝覆盖其他位置。运行结果被 Git 忽略。

## 输出与失败处理

- `predictions.jsonl`：统一预测记录；保留 API 原始响应、原始概率、归一化概率、模型版本、usage 和延迟。
- `errors.jsonl`：逐题调用失败记录；有失败时命令返回非零状态。
- `run_metadata.json`：模型、输入路径、请求数和成功/失败数。

概率 key 与请求选项不一致、未知答案 key、无效概率等情况会作为错误记录，不会静默丢弃。成功记录才进入分析。现在每题发一条 Choice 请求；self-report 与多次采样尚未接入，需要作为额外、有明确预算的实验条件增加。

## 当前边界

- 这是 Jev 的 API 接入点，不是完整实验适配层。
- 没有自动重试、并发调度、断点续跑或调用成本折算；正式大规模运行前应补上并测试。
- `jev-1.13.0` 是为匹配原 proposal 选择的版本。正式调用前需确认账号当前可用版本；将实际响应中的 `model` 字段作为运行证据。
- Qwen3 仍需另行实现本地推理 adapter；本文件只接 Jev。
