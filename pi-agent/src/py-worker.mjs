/**
 * 常驻 Python worker 的客户端与进程池。
 *
 * 为什么要池（见 PI_MIGRATION_PLAN.md §8 决策 1）：
 *   探索的同一层会并发执行多个问题，而单个 worker 的行分隔 JSON 协议是**串行**的
 *   （一个请求一个响应，靠顺序配对）。若只有一个 worker，并发会退化成排队。
 *
 * 内存实测（InsightBench，500 行）：单 worker 205MB，其中库占 202MB、数据仅 3MB。
 * 所以小表场景下 4 workers 约 820MB，很便宜。大数据集（DataGovBench 21 万行 × 5 表）
 * 需靠共享 SQLite 文件控制内存——现有 bridge 本就用 tempfile 而非 :memory:。
 */
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

/** 单个常驻 worker 进程。请求串行排队，保证行协议的请求/响应配对。 */
export class PyWorker {
  #proc;
  #rl;
  #pending = new Map(); // id -> {resolve, reject}
  #seq = 0;
  #ready;
  #closed = false;

  constructor({ python, workerScript, env = {} }) {
    this.#proc = spawn(python, ["-u", workerScript], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, ...env },
    });

    this.#rl = createInterface({ input: this.#proc.stdout });
    this.#rl.on("line", (line) => this.#onLine(line));

    // worker 就绪信号走 stderr（stdout 是协议通道，不能污染）
    this.#ready = new Promise((resolve, reject) => {
      const onErr = (chunk) => {
        const s = chunk.toString();
        if (s.includes("WORKER_READY")) {
          this.#proc.stderr.off("data", onErr);
          resolve();
        }
      };
      this.#proc.stderr.on("data", onErr);
      this.#proc.once("error", reject);
      this.#proc.once("exit", (code) => {
        if (code !== 0 && code !== null) reject(new Error(`worker exited ${code}`));
      });
    });

    // 进程意外退出时，让所有在途请求失败，而不是永久挂起
    this.#proc.once("exit", (code, signal) => {
      this.#closed = true;
      const err = new Error(`python worker exited (code=${code}, signal=${signal})`);
      for (const { reject } of this.#pending.values()) reject(err);
      this.#pending.clear();
    });
  }

  #onLine(line) {
    if (!line.trim()) return;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      // 协议外的输出（理论上不该有）；记录但不崩
      console.error("[PyWorker] non-JSON on stdout:", line.slice(0, 200));
      return;
    }
    const p = this.#pending.get(msg.id);
    if (!p) return;
    this.#pending.delete(msg.id);
    if (msg.ok) p.resolve(msg.result);
    else p.reject(new Error(msg.error + (msg.traceback ? `\n${msg.traceback}` : "")));
  }

  get ready() {
    return this.#ready;
  }

  async call(op, args = {}) {
    if (this.#closed) throw new Error("worker is closed");
    await this.#ready;
    const id = `r${++this.#seq}`;
    const promise = new Promise((resolve, reject) =>
      this.#pending.set(id, { resolve, reject }),
    );
    this.#proc.stdin.write(JSON.stringify({ id, op, args }) + "\n");
    return promise;
  }

  async close() {
    if (this.#closed) return;
    try {
      await this.call("close");
    } catch {
      /* 关闭途中出错无所谓 */
    }
    this.#closed = true;
    this.#proc.stdin.end();
    this.#proc.kill();
  }
}

/**
 * worker 池：把并发请求分发到多个 worker。
 *
 * 注意：每个 worker 各自持有一个 bridge，因此**每个 worker 都要 load_csv**。
 * `loadAll()` 负责这件事，之后任意 worker 都能处理该数据集的查询。
 */
export class PyWorkerPool {
  #workers = [];
  #idle = [];
  #queue = [];

  constructor({ python, workerScript, size = 4, env = {} }) {
    for (let i = 0; i < size; i++) {
      const w = new PyWorker({ python, workerScript, env });
      this.#workers.push(w);
      this.#idle.push(w);
    }
  }

  get size() {
    return this.#workers.length;
  }

  async ready() {
    await Promise.all(this.#workers.map((w) => w.ready));
  }

  /** 所有 worker 载入同一数据集；返回第一个 worker 的 schema（各 worker 一致）。 */
  async loadAll(args) {
    const results = await Promise.all(
      this.#workers.map((w) => w.call("load_csv", args)),
    );
    return results[0];
  }

  /**
   * 预热每个 worker 的 Python 沙箱。
   *
   * `execute_python_from_sql` 会在沙箱 local_vars 里预注入 numpy/pandas/scipy/
   * sklearn/statsmodels/… ——首次调用要为此付约 1 秒。实测不预热时
   * 「3 个 1 秒任务并发」耗时 5.5s（每个 worker 各自摊到 import 成本）；
   * 预热后同样负载 1.01s，即真并行。
   *
   * 把这笔一次性开销挪到池初始化阶段，避免污染首批探索问题的耗时。
   */
  async warmup() {
    await Promise.all(
      this.#workers.map((w) =>
        w.call("python", { sql: "SELECT 1 AS x", code: "pass" }).catch(() => {}),
      ),
    );
  }

  /** 借一个空闲 worker 执行，用完归还。并发超过池大小时排队。 */
  async call(op, args = {}) {
    const w = await this.#acquire();
    try {
      return await w.call(op, args);
    } finally {
      this.#release(w);
    }
  }

  #acquire() {
    const w = this.#idle.pop();
    if (w) return Promise.resolve(w);
    return new Promise((resolve) => this.#queue.push(resolve));
  }

  #release(w) {
    const next = this.#queue.shift();
    if (next) next(w);
    else this.#idle.push(w);
  }

  async close() {
    await Promise.all(this.#workers.map((w) => w.close()));
    this.#workers = [];
    this.#idle = [];
  }
}
