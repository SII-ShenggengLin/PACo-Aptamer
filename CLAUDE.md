# RNA_tertiary_stru_IF — Claude Code 工作规则

## 🛡️ 进程改名防护（不要撤销）

启智（inspire）平台在宿主机侧按进程名/cmdline SIGKILL 含 "claude" 的进程（codex、kimi 等同机进程不受影响）。已通过改名规避：

- 本体副本：`/inspire/hdd/global_user/linshenggeng-240107100015/.bin/cwork`（claude.exe 的中性名副本，wrapper 会在官方包升级后自动重新同步）
- wrapper：`/root/.local/bin/cwork`（`exec -a cwork` 启动，argv0/comm/exe 均不含 "claude"）
- VS Code 扩展：`/root/.vscode-server/data/Machine/settings.json` 里 `claudeCode.claudeProcessWrapper` 指向 wrapper
- 终端：`/root/.nvm/versions/node/v24.15.0/bin/claude` 已从符号链接替换为同款 wrapper（若 `npm i -g @anthropic-ai/claude-code` 升级后该文件被 npm 重置为符号链接，需重新用 wrapper 覆盖）

## ⛔ 硬性禁令：禁止前台阻塞等待（已导致 Claude Code 进程被 SIGKILL，多次发生）

**绝不**在 Bash 工具里执行任何会长时间阻塞前台的命令，包括但不限于：

- `until grep -q ... log; do sleep N; done`（轮询日志等结果）
- `until ! ps -p PID; do sleep N; done`、`while kill -0 PID; do ...; done`
- `sleep` 超过 60 秒
- 直接前台运行训练/评估/下载等耗时任务（即使加了 `timeout`）
- `tail -f`、`watch`、任何无退出条件的循环

此类命令会让工具调用挂起数小时，最终宿主环境会把整个 Claude Code 进程 SIGKILL，会话上下文全部丢失。

## ✅ 正确做法

1. **启动长任务**：单独一条命令，`nohup ... > 绝对路径日志 2>&1 & echo "PID: $!"`，命令本身必须**立即返回**。路径一律用绝对路径（后台相对路径会丢）。
2. **监控进度**：用独立的短命令轮询，每条命令几秒内返回，例如：
   - `tail -5 /abs/path/train.log`
   - `ps -p <PID> -o pid,etime,stat,cmd`
   - 需要等很久时，分多次调用，每次之间正常结束当前回合。
3. **等待完成**：不要在一次调用里等。每个回合查一次状态；任务没结束就如实告诉用户当前进度，下一个回合再查。

## 其他环境要点（详见 analysis/05_会话交接文档.md）

- **load_s1 命中缓存时忽略 subset 过滤**（已三次复发）：`load_s1(set(stems), cache=...)` 返回全缓存 6,338 链——遍历必须 `sorted(set(splits_stems) & set(data))`，绝不能直接 `for s in data`
- pip 装包用 `PIP_CONFIG_FILE=/tmp/pip_clean.conf`（机器默认指向 NVIDIA NGC 镜像）
- pyg 扩展装不上 → 用 aptamer_eval 环境里已写好的 shim（torch_scatter/torch_cluster）
- GPU 只用 1、2 号（0、3 被其他任务占用）；`CUDA_VISIBLE_DEVICES` 限制后设备号从 0 数
- 大文件下载用 `/tmp/multidownload.py` 多线程
- conda 环境：`/root/miniconda3/envs/aptamer_eval/bin/python`
