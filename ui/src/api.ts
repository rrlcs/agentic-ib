import type {
  JobResult,
  MetricsResponse,
  TaskMetrics,
  TraceEvent,
} from "./types";

const RAW_API_BASE = (import.meta.env.VITE_API_BASE || "").trim();
export const API_BASE =
  RAW_API_BASE || `${window.location.protocol}//${window.location.hostname}:8000`;

export async function postChat(
  message: string,
  options: { model?: string; company?: string; symbol?: string } = {}
): Promise<{ task_id: string }> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, ...options }),
  });
  if (!response.ok) throw new Error(`chat failed (${response.status})`);
  return response.json();
}

export async function fetchJobResult(taskId: string): Promise<JobResult | null> {
  const response = await fetch(`${API_BASE}/job/${encodeURIComponent(taskId)}`);
  if (!response.ok) return null;
  const body = (await response.json()) as JobResult;
  if (body.status === "not_found") return null;
  return body;
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const response = await fetch(`${API_BASE}/metrics`);
  if (!response.ok) return {};
  return (await response.json()) as MetricsResponse;
}

export async function fetchTaskMetrics(taskId: string): Promise<TaskMetrics | null> {
  const response = await fetch(`${API_BASE}/metrics/${encodeURIComponent(taskId)}`);
  if (!response.ok) return null;
  const body = (await response.json()) as TaskMetrics;
  if ("status" in body && body.status === "not_found") return null;
  return body;
}

export interface TraceStream {
  close: () => void;
}

const TRACED_EVENTS = [
  "router_planned",
  "agent_handoff",
  "research_started",
  "research_completed",
  "financial_started",
  "financial_completed",
  "risk_started",
  "risk_completed",
  "synthesis_started",
  "synthesis_completed",
  "validator_started",
  "validator_decided",
  "answer_started",
  "answer_completed",
  "action_started",
  "action_completed",
  "action_memo_resolved",
  "feedback_loop_invoked",
  "pipeline_complete",
  "llm_call_started",
  "llm_call_completed",
  "llm_token",
  "parallel_started",
  "parallel_completed",
  "financial_api_request_started",
  "financial_api_request_completed",
  "memo_stored",
  "memo_store_failed",
  "memo_retrieve_failed",
  "mcp_envelope_received",
  "agent_loop_started",
  "agent_loop_completed",
  "agent_iteration",
  "agent_iteration_final",
  "agent_tool_call",
  "agent_tool_result",
  "web_search_started",
  "web_search_completed",
  "vector_search_completed",
  "paper_trade_submitted",
  "order_status_fetched",
];

export function streamTraces(
  taskId: string,
  onEvent: (event: TraceEvent) => void,
  onComplete?: () => void
): TraceStream {
  const url = `${API_BASE}/stream/${encodeURIComponent(taskId)}`;
  const source = new EventSource(url);

  const handleData = (raw: string) => {
    try {
      const parsed = JSON.parse(raw) as TraceEvent;
      onEvent(parsed);
      if (parsed.event === "pipeline_complete") {
        onComplete?.();
        source.close();
      }
    } catch {
      // ignore malformed
    }
  };

  source.onmessage = (msg: MessageEvent) => handleData(msg.data);
  source.addEventListener("ping", () => {
    /* keep-alive */
  });
  for (const evt of TRACED_EVENTS) {
    source.addEventListener(evt, (msg) => handleData((msg as MessageEvent).data));
  }
  source.onerror = () => {
    // Browser auto-reconnects; rely on pipeline_complete to close.
  };
  return { close: () => source.close() };
}
