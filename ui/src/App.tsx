import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchJobResult,
  fetchMetrics,
  fetchTaskMetrics,
  postChat,
  streamTraces,
} from "./api";
import type {
  ChatMessage,
  JobResult,
  MetricsResponse,
  PipelineNode,
  PipelineStep,
  TaskMetrics,
  TraceEvent,
} from "./types";

type SidePanelMode = "pipeline" | "activity" | "result" | "metrics";

interface TaskState {
  taskId: string;
  traces: TraceEvent[];
  pipeline: PipelineStep[];
  result?: JobResult;
  intent?: string;
  model?: string;
  metrics?: TaskMetrics;
}

const MODEL_CHOICES = [
  { id: "gpt-4o-mini", label: "GPT-4o mini · fast" },
  { id: "gpt-4o", label: "GPT-4o · smart" },
  { id: "gpt-4-turbo", label: "GPT-4 Turbo" },
  { id: "gpt-3.5-turbo", label: "GPT-3.5 Turbo" },
];
const DEFAULT_MODEL = MODEL_CHOICES[0].id;

const STORAGE_KEYS = {
  messages: "agentic.messages.v1",
  model: "agentic.model.v1",
  tasks: "agentic.tasks.v1",
};

const INTRO_MESSAGE: ChatMessage = {
  id: "intro",
  role: "assistant",
  text:
    "Ask me about a company. Examples:\n• Should I invest in NVIDIA?\n• What does Tesla do?\n• Risks of investing in Apple?\n• Place a paper trade for AAPL",
};

export function App() {
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadMessages());
  const [input, setInput] = useState("");
  const [tasks, setTasks] = useState<Record<string, TaskState>>(() => loadTasks());
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [panelMode, setPanelMode] = useState<SidePanelMode>("pipeline");
  const [metrics, setMetrics] = useState<MetricsResponse>({});
  const [model, setModel] = useState<string>(() => loadModel());

  const chatScrollRef = useRef<HTMLDivElement>(null);
  const wasNearBottomRef = useRef(true);
  const placeholderIdRef = useRef<string | null>(null);

  // Persist messages, tasks, and model.
  useEffect(() => {
    try {
      const trimmed = messages.slice(-50);
      localStorage.setItem(STORAGE_KEYS.messages, JSON.stringify(trimmed));
    } catch {
      /* quota / disabled */
    }
  }, [messages]);
  useEffect(() => {
    try {
      const compact: Record<string, TaskState> = {};
      for (const [id, t] of Object.entries(tasks) as [string, TaskState][]) {
        compact[id] = { ...t, traces: t.traces.slice(-50) };
      }
      localStorage.setItem(STORAGE_KEYS.tasks, JSON.stringify(compact));
    } catch {
      /* quota / disabled */
    }
  }, [tasks]);
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEYS.model, model);
    } catch {
      /* ignore */
    }
  }, [model]);

  // Track whether user is near the bottom (so we don't yank them away while reading history).
  const handleChatScroll = useCallback(() => {
    const el = chatScrollRef.current;
    if (!el) return;
    wasNearBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 140;
  }, []);

  useEffect(() => {
    if (!chatScrollRef.current) return;
    if (wasNearBottomRef.current) {
      chatScrollRef.current.scrollTo({
        top: chatScrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages]);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      const data = await fetchMetrics();
      if (!cancelled) setMetrics(data);
    };
    tick();
    const id = setInterval(tick, 4000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // Poll per-task metrics for the active task while it runs.
  useEffect(() => {
    if (!activeTaskId) return;
    let cancelled = false;
    const tick = async () => {
      const data = await fetchTaskMetrics(activeTaskId);
      if (cancelled || !data) return;
      setTasks((prev) => {
        const current = prev[activeTaskId];
        if (!current) return prev;
        return { ...prev, [activeTaskId]: { ...current, metrics: data } };
      });
    };
    tick();
    const id = setInterval(tick, 1500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [activeTaskId]);

  const sending = useMemo(() => {
    if (!activeTaskId) return false;
    const task = tasks[activeTaskId];
    return task ? !task.result : false;
  }, [activeTaskId, tasks]);

  const handleClear = useCallback(() => {
    setMessages([INTRO_MESSAGE]);
    setTasks({});
    setActiveTaskId(null);
    placeholderIdRef.current = null;
  }, []);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending) return;
    setInput("");
    wasNearBottomRef.current = true; // user just sent, snap to bottom
    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      text,
    };
    const placeholder: ChatMessage = {
      id: `a-${Date.now()}`,
      role: "assistant",
      text: "",
      thinking: true,
      model,
    };
    placeholderIdRef.current = placeholder.id;
    setMessages((prev) => [...prev, userMsg, placeholder]);

    let taskId: string;
    try {
      const result = await postChat(text, { model });
      taskId = result.task_id;
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === placeholder.id
            ? { ...m, thinking: false, text: `Error: ${(err as Error).message}` }
            : m
        )
      );
      return;
    }

    setMessages((prev) =>
      prev.map((m) => (m.id === placeholder.id ? { ...m, taskId } : m))
    );
    setActiveTaskId(taskId);
    setPanelMode("pipeline");
    setTasks((prev) => ({
      ...prev,
      [taskId]: { taskId, traces: [], pipeline: [], model },
    }));

    const stream = streamTraces(
      taskId,
      (event) => {
        setTasks((prev) => {
          const current = prev[taskId] ?? {
            taskId,
            traces: [],
            pipeline: [],
            model,
          };
          let nextPipeline = current.pipeline;
          let nextIntent = current.intent;
          if (event.event === "router_planned" && Array.isArray(event.pipeline)) {
            nextPipeline = event.pipeline as PipelineStep[];
            nextIntent = (event.intent as string) ?? current.intent;
          }
          return {
            ...prev,
            [taskId]: {
              ...current,
              traces: [...current.traces, event],
              pipeline: nextPipeline,
              intent: nextIntent,
            },
          };
        });

        if (
          event.event === "llm_token" &&
          typeof event.delta === "string" &&
          event.agent === "answer_agent" &&
          placeholderIdRef.current
        ) {
          const delta = event.delta;
          const pid = placeholderIdRef.current;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === pid
                ? {
                    ...m,
                    thinking: false,
                    streaming: true,
                    text: (m.text ?? "") + delta,
                  }
                : m
            )
          );
        }
      },
      async () => {
        const result = await fetchJobResult(taskId);
        if (result) {
          setTasks((prev) => {
            const current = prev[taskId] ?? {
              taskId,
              traces: [],
              pipeline: [],
              model,
            };
            return { ...prev, [taskId]: { ...current, result } };
          });
          const pid = placeholderIdRef.current;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === pid
                ? {
                    ...m,
                    thinking: false,
                    streaming: false,
                    taskId,
                    text:
                      m.text && m.text.length > 0
                        ? m.text
                        : formatAssistantReply(result),
                  }
                : m
            )
          );
        } else {
          const pid = placeholderIdRef.current;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === pid
                ? {
                    ...m,
                    thinking: false,
                    streaming: false,
                    text: "Pipeline finished but no result was found.",
                  }
                : m
            )
          );
        }
        placeholderIdRef.current = null;
      }
    );
    return () => stream.close();
  };

  const activeTask = activeTaskId ? tasks[activeTaskId] : undefined;

  return (
    <div className="app">
      <section className="left">
        <div className="header">
          <h1>Agentic IB Chat</h1>
          <span className="meta">
            {activeTask
              ? `task: ${activeTask.taskId.slice(0, 8)}…`
              : "ready"}
            <button className="text-btn" onClick={handleClear}>
              clear
            </button>
          </span>
        </div>
        <div className="chat" ref={chatScrollRef} onScroll={handleChatScroll}>
          {messages.map((m) => (
            <div
              key={m.id}
              className={`bubble ${m.role}${m.thinking ? " thinking" : ""}${
                m.streaming ? " streaming" : ""
              }`}
              onClick={() => m.taskId && setActiveTaskId(m.taskId)}
              style={{ cursor: m.taskId ? "pointer" : "default" }}
            >
              {m.thinking && !m.text ? <ThinkingDots /> : m.text}
              {m.role === "assistant" && m.model && (
                <div className="bubble-meta">{m.model}</div>
              )}
            </div>
          ))}
        </div>
        <div className="composer">
          <select
            className="model-picker"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            title="Choose the model used for the next message"
          >
            {MODEL_CHOICES.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          <input
            placeholder="Ask about a company…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            disabled={sending}
          />
          <button onClick={handleSend} disabled={sending || !input.trim()}>
            Send
          </button>
        </div>
      </section>

      <section className="right">
        <div className="header">
          <h1>Observability</h1>
          <span className="meta">
            {Object.keys(tasks).length} task{Object.keys(tasks).length === 1 ? "" : "s"}
          </span>
        </div>
        <div className="tabs">
          {(
            [
              ["pipeline", "Pipeline"],
              ["activity", "Activity"],
              ["result", "Result"],
              ["metrics", "Metrics"],
            ] as [SidePanelMode, string][]
          ).map(([mode, label]) => (
            <div
              key={mode}
              className={`tab ${panelMode === mode ? "active" : ""}`}
              onClick={() => setPanelMode(mode)}
            >
              {label}
            </div>
          ))}
        </div>
        <div className="panel">
          {panelMode === "pipeline" ? (
            <PipelineView task={activeTask} />
          ) : panelMode === "activity" ? (
            <ActivityView task={activeTask} />
          ) : panelMode === "result" ? (
            <ResultView task={activeTask} />
          ) : (
            <MetricsView task={activeTask} global={metrics} />
          )}
        </div>
      </section>
    </div>
  );
}

function ThinkingDots() {
  return (
    <span className="dots">
      <span /> <span /> <span />
    </span>
  );
}

function PipelineView({ task }: { task?: TaskState }) {
  if (!task || task.pipeline.length === 0) {
    return (
      <div className="empty">Send a message to see the agent pipeline.</div>
    );
  }
  const nodes = computePipelineNodes(task.pipeline, task.traces);
  const validator = task.traces.find((e) => e.event === "validator_decided");
  const feedback = task.traces.filter((e) => e.event === "feedback_loop_invoked");

  return (
    <div className="state-machine">
      <div className="sm-header">
        <span className="pill accent">{task.intent ?? "intent: ?"}</span>
        <span className="pill">{task.model ?? "default model"}</span>
        {feedback.length > 0 && (
          <span className="pill warn">↺ feedback × {feedback.length}</span>
        )}
        {validator?.decision && (
          <span
            className={`pill ${
              validator.decision === "accept" ? "good" : "warn"
            }`}
          >
            validator: {String(validator.decision)}
          </span>
        )}
      </div>

      <div className="sm-nodes">
        {nodes.map((node, idx) => (
          <div key={node.id} className="sm-node-row">
            <div className={`sm-node sm-${node.status} ${node.parallel ? "sm-parallel" : ""}`}>
              <div className="sm-title">
                {node.parallel ? "Parallel" : agentLabel(node.agents[0])}
              </div>
              {node.parallel && (
                <div className="sm-children">
                  {node.agents.map((a) => (
                    <div key={a} className="sm-child">
                      {agentLabel(a)}
                    </div>
                  ))}
                </div>
              )}
              {node.detail && (
                <div className="sm-detail">{node.detail}</div>
              )}
              {node.retries ? (
                <div className="sm-detail">retries: {node.retries}</div>
              ) : null}
              {node.iterations && node.iterations > 1 ? (
                <div className="sm-detail">iterations: {node.iterations}</div>
              ) : null}
              {node.tool_calls && node.tool_calls.length > 0 && (
                <div className="sm-tools">
                  {node.tool_calls.map((tc, i) => (
                    <div key={i} className="sm-tool">
                      <span className="sm-tool-name">→ {tc.tool}</span>
                      {tc.args_preview && (
                        <span className="sm-tool-args">
                          {truncate(tc.args_preview, 80)}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
            {idx !== nodes.length - 1 && <div className="sm-arrow">↓</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

function ActivityView({ task }: { task?: TaskState }) {
  if (!task) {
    return <div className="empty">Send a message to see activity.</div>;
  }
  if (task.traces.length === 0) {
    return <div className="empty">Waiting for trace events…</div>;
  }
  const interesting = task.traces.filter((e) => e.event !== "llm_token" && e.event !== "ping");
  return (
    <>
      {interesting.map((event, idx) => (
        <div key={`${event.ts}-${idx}`} className={`trace event-${event.event}`}>
          <div className="row">
            <span className="name">{event.event}</span>
            {event.agent && <span className="pill">{String(event.agent)}</span>}
            <span className="pill mono">
              {new Date((event.ts as number) * 1000).toLocaleTimeString()}
            </span>
          </div>
          <div className="row" style={{ marginTop: 4, color: "var(--muted)" }}>
            {Object.entries(event)
              .filter(
                ([k]) =>
                  !["ts", "event", "task_id", "component", "agent"].includes(k)
              )
              .map(([k, v]) => (
                <span key={k} className="pill">
                  {k}={truncate(stringify(v))}
                </span>
              ))}
          </div>
        </div>
      ))}
    </>
  );
}

function ResultView({ task }: { task?: TaskState }) {
  if (!task) return <div className="empty">No active task.</div>;
  if (!task.result) return <div className="empty">Pipeline running…</div>;
  const r = task.result;
  return (
    <div className="result-card">
      <div className="result-row">
        <span className="pill accent">{r.intent ?? "intent"}</span>
        <span className={`pill ${r.status === "success" ? "good" : "warn"}`}>
          {r.status ?? "?"}
        </span>
        {r.memo_source && (
          <span className="pill">memo: {r.memo_source}</span>
        )}
      </div>
      {r.answer && (
        <div className="result-section">
          <div className="result-label">Chat reply</div>
          <div className="result-text">{r.answer}</div>
        </div>
      )}
      {r.action ? <OrderVerification action={r.action as Record<string, unknown>} /> : null}
      {r.result && (
        <div className="result-section">
          <div className="result-label">Memo</div>
          <div className="result-text">{r.result}</div>
        </div>
      )}
      {r.validation && (
        <div className="result-section">
          <div className="result-label">Validation</div>
          <pre className="result-pre">{JSON.stringify(r.validation, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

function OrderVerification({ action }: { action: Record<string, unknown> }) {
  const executed = Boolean(action.executed);
  const status = String(action.broker_status ?? action.status ?? "—");
  const orderId = String(action.order_id ?? "");
  const filledQty = action.filled_qty;
  const filledPrice = action.filled_avg_price;
  const dashboard = action.dashboard_url ? String(action.dashboard_url) : null;
  const verdict = executed
    ? "✓ Order executed"
    : status === "skipped"
    ? "skipped (no broker creds)"
    : status === "dry_run"
    ? "dry-run only"
    : status === "timeout"
    ? "submitted, no fill within timeout"
    : "not executed";
  return (
    <div className="result-section order-card">
      <div className="result-label">Paper trade</div>
      <div className={`order-verdict order-${executed ? "ok" : "warn"}`}>{verdict}</div>
      <div className="order-grid">
        <Field label="action" value={String(action.action ?? "—")} />
        <Field label="qty" value={String(action.quantity ?? "—")} />
        <Field label="status" value={status} />
        <Field
          label="filled"
          value={
            filledQty != null
              ? `${filledQty}${filledPrice != null ? ` @ $${Number(filledPrice).toFixed(2)}` : ""}`
              : "—"
          }
        />
        <Field label="order id" value={orderId ? truncate(orderId, 14) : "—"} />
      </div>
      {String(action.rationale ?? "") && (
        <div className="order-rationale">{String(action.rationale)}</div>
      )}
      {dashboard && (
        <a className="order-link" href={dashboard} target="_blank" rel="noreferrer">
          Open Alpaca paper dashboard ↗
        </a>
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="order-field">
      <div className="order-field-label">{label}</div>
      <div className="order-field-value">{value}</div>
    </div>
  );
}

function MetricsView({
  task,
  global,
}: {
  task?: TaskState;
  global: MetricsResponse;
}) {
  return (
    <div className="metrics-stack">
      <div className="metrics-section">
        <div className="metrics-section-title">This run</div>
        <CurrentRunMetrics task={task} />
      </div>
      <div className="metrics-section">
        <div className="metrics-section-title">All-time</div>
        <GlobalMetrics metrics={global} />
      </div>
    </div>
  );
}

function CurrentRunMetrics({ task }: { task?: TaskState }) {
  if (!task) {
    return <div className="empty">Send a message to see this run's metrics.</div>;
  }
  const m = task.metrics;
  const cards: { label: string; value: string }[] = [
    {
      label: "latency",
      value: m?.latency_ms != null ? `${(m.latency_ms / 1000).toFixed(1)} s` : "running…",
    },
    { label: "LLM calls", value: String(m?.llm_calls ?? 0) },
    { label: "tool calls", value: String(m?.tool_calls ?? 0) },
    { label: "iterations", value: String(m?.agent_iterations ?? 0) },
    { label: "feedback loops", value: String(m?.feedback_loops ?? 0) },
    {
      label: "tokens (in / out)",
      value: `${m?.tokens_in ?? 0} / ${m?.tokens_out ?? 0}`,
    },
  ];
  return (
    <div className="metrics">
      {cards.map((c) => (
        <div key={c.label} className="metric">
          <div className="label">{c.label}</div>
          <div className="value">{c.value}</div>
        </div>
      ))}
    </div>
  );
}

function GlobalMetrics({ metrics }: { metrics: MetricsResponse }) {
  const cards: { label: string; value: string }[] = [
    {
      label: "avg latency",
      value: metrics.latency_ms_avg != null ? `${(metrics.latency_ms_avg / 1000).toFixed(1)} s` : "—",
    },
    {
      label: "p95 latency",
      value: metrics.latency_ms_p95 != null ? `${(metrics.latency_ms_p95 / 1000).toFixed(1)} s` : "—",
    },
    {
      label: "tasks (ok / total)",
      value: `${metrics.tasks_succeeded ?? 0} / ${metrics.tasks_total ?? 0}`,
    },
    { label: "success rate", value: successRate(metrics) },
  ];
  return (
    <div className="metrics">
      {cards.map((c) => (
        <div key={c.label} className="metric small">
          <div className="label">{c.label}</div>
          <div className="value">{c.value}</div>
        </div>
      ))}
    </div>
  );
}

function successRate(m: MetricsResponse): string {
  const total = m.tasks_total ?? 0;
  if (!total) return "—";
  const ok = m.tasks_succeeded ?? 0;
  return `${Math.round((ok / total) * 100)}%`;
}

function truncate(s: string, max = 40): string {
  if (s.length <= max) return s;
  return `${s.slice(0, Math.max(1, max - 3))}…`;
}

function stringify(v: unknown): string {
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

function formatAssistantReply(result: JobResult): string {
  if (!result) return "(no result)";
  if (result.status === "failed") {
    return `Pipeline failed: ${result.reason ?? "unknown"}`;
  }
  if (result.answer) return result.answer;
  if (result.result) return result.result;
  if (result.intent) return `Intent: ${result.intent}\n\n${JSON.stringify(result, null, 2)}`;
  return JSON.stringify(result, null, 2);
}

function agentLabel(agent: string): string {
  const map: Record<string, string> = {
    research_agent: "Research",
    financial_agent: "Financials",
    risk_agent: "Risk",
    synthesis_agent: "Synthesis",
    validator_agent: "Validator",
    answer_agent: "Answer",
    action_agent: "Trade Action",
    router: "Router",
  };
  return map[agent] ?? agent;
}

function computePipelineNodes(
  pipeline: PipelineStep[],
  traces: TraceEvent[]
): PipelineNode[] {
  const status = new Map<string, "running" | "done" | "failed" | "feedback">();
  const startedAgents = new Set<string>();
  const completedAgents = new Set<string>();
  const detail = new Map<string, string>();
  const retries = new Map<string, number>();
  const toolCalls = new Map<string, { agent: string; tool: string; args_preview?: string; result_preview?: string }[]>();
  const iterations = new Map<string, number>();
  const lastToolPerAgent = new Map<string, number>(); // index of last tool_call entry

  for (const ev of traces) {
    const agent = guessAgentFromEvent(ev);
    if (ev.event.endsWith("_started") && agent) {
      startedAgents.add(agent);
      status.set(agent, "running");
    }
    if (ev.event.endsWith("_completed") && agent) {
      completedAgents.add(agent);
      if (status.get(agent) !== "feedback") status.set(agent, "done");
    }
    if (ev.event === "validator_decided" && ev.decision === "accept") {
      status.set("validator_agent", "done");
    }
    if (ev.event === "validator_decided" && ev.decision && ev.decision !== "accept") {
      detail.set("validator_agent", `decision: ${ev.decision}`);
      status.set("validator_agent", "feedback");
    }
    if (ev.event === "feedback_loop_invoked" && typeof ev.target_agent === "string") {
      const target = ev.target_agent;
      retries.set(target, (retries.get(target) ?? 0) + 1);
      status.set(target, "feedback");
    }
    if (ev.event === "action_memo_resolved") {
      detail.set("action_agent", `memo: ${ev.source}`);
    }
    if (ev.event === "validator_decided") {
      detail.set("validator_agent", `${ev.decision} · conf=${ev.confidence ?? "?"}`);
    }
    if (ev.event === "agent_iteration" && typeof ev.agent === "string") {
      iterations.set(ev.agent, Number(ev.iteration ?? 0));
    }
    if (ev.event === "agent_tool_call" && typeof ev.agent === "string" && typeof ev.tool === "string") {
      const list = toolCalls.get(ev.agent) ?? [];
      list.push({
        agent: ev.agent,
        tool: String(ev.tool),
        args_preview: ev.args_preview ? String(ev.args_preview) : undefined,
      });
      lastToolPerAgent.set(ev.agent, list.length - 1);
      toolCalls.set(ev.agent, list);
    }
    if (ev.event === "agent_tool_result" && typeof ev.agent === "string") {
      const list = toolCalls.get(ev.agent);
      const last = lastToolPerAgent.get(ev.agent);
      if (list && last !== undefined && list[last]) {
        list[last].result_preview = `${ev.result_chars ?? "?"} chars`;
      }
    }
  }

  return pipeline.map((step, idx) => {
    const agents = Array.isArray(step) ? step : [step];
    const parallel = Array.isArray(step);
    const id = `${idx}-${agents.join("+")}`;
    const allDone = agents.every((a) => completedAgents.has(a));
    const anyRunning = agents.some((a) => startedAgents.has(a) && !completedAgents.has(a));
    const anyFeedback = agents.some((a) => status.get(a) === "feedback");
    let s: PipelineNode["status"] = "pending";
    if (anyFeedback) s = "feedback";
    else if (allDone) s = "done";
    else if (anyRunning) s = "running";

    const detailStrings = agents
      .map((a) => detail.get(a))
      .filter((x): x is string => Boolean(x));
    const retryTotal = agents.reduce((sum, a) => sum + (retries.get(a) ?? 0), 0);
    const calls = agents.flatMap((a) => toolCalls.get(a) ?? []);
    const iterTotal = agents.reduce((sum, a) => sum + (iterations.get(a) ?? 0), 0);
    return {
      id,
      agents,
      parallel,
      status: s,
      detail: detailStrings.join(" · ") || undefined,
      retries: retryTotal || undefined,
      tool_calls: calls.length ? calls : undefined,
      iterations: iterTotal || undefined,
    };
  });
}

function guessAgentFromEvent(event: TraceEvent): string | null {
  const e = event.event;
  if (e === "llm_token" || e === "llm_call_started" || e === "llm_call_completed") {
    return null;
  }
  const lookup: Record<string, string> = {
    research_started: "research_agent",
    research_completed: "research_agent",
    financial_started: "financial_agent",
    financial_completed: "financial_agent",
    risk_started: "risk_agent",
    risk_completed: "risk_agent",
    synthesis_started: "synthesis_agent",
    synthesis_completed: "synthesis_agent",
    validator_started: "validator_agent",
    validator_decided: "validator_agent",
    answer_started: "answer_agent",
    answer_completed: "answer_agent",
    action_started: "action_agent",
    action_completed: "action_agent",
  };
  return lookup[e] ?? null;
}

function loadMessages(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.messages);
    if (!raw) return [INTRO_MESSAGE];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length === 0) return [INTRO_MESSAGE];
    return parsed as ChatMessage[];
  } catch {
    return [INTRO_MESSAGE];
  }
}

function loadTasks(): Record<string, TaskState> {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.tasks);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return typeof parsed === "object" && parsed ? parsed : {};
  } catch {
    return {};
  }
}

function loadModel(): string {
  try {
    return localStorage.getItem(STORAGE_KEYS.model) || DEFAULT_MODEL;
  } catch {
    return DEFAULT_MODEL;
  }
}
