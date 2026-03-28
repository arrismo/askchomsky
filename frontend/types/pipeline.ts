export type StageStatus = "idle" | "running" | "done" | "error";

export interface StageEvent {
  id: string;
  label: string;
  status: StageStatus;
  detail?: string;
  attempt?: number;
}

export interface DoneEvent {
  answer: string;
}

export interface ErrorEvent {
  message: string;
}

// The fixed set of node IDs in pipeline order
export const PIPELINE_STAGES = [
  "intent",
  "rewrite",
  "rag_init",
  "retrieval_1",
  "retrieval_2",
  "retrieval_3",
  "citations",
  "verification",
  "answer",
] as const;

export type PipelineStageId = (typeof PIPELINE_STAGES)[number];

export interface NodeState {
  id: string;
  label: string;
  status: StageStatus;
  detail: string;
}
