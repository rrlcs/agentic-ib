export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
  taskId?: string;
  thinking?: boolean;
  streaming?: boolean;
  model?: string;
}

export interface TraceEvent {
  ts: number;
  event: string;
  task_id: string;
  component?: string;
  agent?: string;
  delta?: string;
  pipeline?: PipelineStep[];
  intent?: string;
  decision?: string;
  next_agent?: string;
  agents?: string[] | Record<string, string>;
  [key: string]: unknown;
}

export type PipelineStep = string | string[];

export interface JobResult {
  task_id?: string;
  status?: string;
  intent?: string;
  answer?: string;
  result?: string;
  validation?: Record<string, unknown> | null;
  action?: Record<string, unknown> | null;
  context_keys?: string[];
  reason?: string;
  decision?: string;
  valid?: boolean;
  grounded?: boolean;
  confidence?: number;
  issues?: string;
  memo_source?: string;
}

export interface MetricsResponse {
  tasks_total?: number;
  tasks_succeeded?: number;
  tasks_failed?: number;
  tasks_in_flight?: number;
  llm_calls_total?: number;
  feedback_loops?: number;
  latency_ms_avg?: number | null;
  latency_ms_p95?: number | null;
  samples?: number;
}

export interface TaskMetrics {
  task_id?: string;
  status?: string;
  llm_calls?: number;
  tool_calls?: number;
  agent_iterations?: number;
  feedback_loops?: number;
  tokens_in?: number;
  tokens_out?: number;
  latency_ms?: number | null;
  started_at?: number | null;
  completed_at?: number | null;
}

export type NodeStatus = "pending" | "running" | "done" | "failed" | "feedback";

export interface AgentToolCall {
  agent: string;
  tool: string;
  args_preview?: string;
  result_preview?: string;
}

export interface PipelineNode {
  id: string;
  agents: string[];
  parallel: boolean;
  status: NodeStatus;
  detail?: string;
  retries?: number;
  tool_calls?: AgentToolCall[];
  iterations?: number;
}
