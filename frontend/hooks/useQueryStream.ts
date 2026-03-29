"use client";

import { useCallback, useRef, useState } from "react";
import type { NodeState, StageEvent } from "@/types/pipeline";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

// Default idle nodes shown before any query is run
const DEFAULT_NODES: NodeState[] = [
  { id: "intent", label: "Intent Router", status: "idle", detail: "" },
  { id: "rewrite", label: "Query Rewrite", status: "idle", detail: "" },
  { id: "rag_init", label: "Loading RAG Store", status: "idle", detail: "" },
  { id: "retrieval_1", label: "Retrieval", status: "idle", detail: "" },
  { id: "retrieval_2", label: "Retrieval (retry)", status: "idle", detail: "" },
  { id: "retrieval_3", label: "Retrieval (retry 2)", status: "idle", detail: "" },
  { id: "citations", label: "Citation Enforcement", status: "idle", detail: "" },
  { id: "verification", label: "Claim Verification", status: "idle", detail: "" },
  { id: "answer", label: "Answer", status: "idle", detail: "" },
];

export function useQueryStream() {
  const [nodes, setNodes] = useState<NodeState[]>(DEFAULT_NODES);
  const [answer, setAnswer] = useState<string>("");
  const [streamingAnswer, setStreamingAnswer] = useState<string>("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string>("");
   const [followUps, setFollowUps] = useState<string[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    setNodes(DEFAULT_NODES.map((n) => ({ ...n, status: "idle", detail: "" })));
    setAnswer("");
    setStreamingAnswer("");
    setError("");
     setFollowUps([]);
  }, []);

  const updateNode = useCallback((event: StageEvent) => {
    setNodes((prev) =>
      prev.map((n) =>
        n.id === event.id
          ? { ...n, label: event.label, status: event.status, detail: event.detail ?? "" }
          : n
      )
    );
  }, []);

  const submit = useCallback(
    async (question: string, mode?: string) => {
      // Cancel any in-flight request
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      reset();
      setStreaming(true);

      try {
        const payload: Record<string, unknown> = { question };
        if (mode) {
          payload.mode = mode;
        }

        const response = await fetch(`${API_URL}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: abortRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          let currentEvent = "";
          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              const raw = line.slice(6).trim();
              try {
                const parsed = JSON.parse(raw);
                if (currentEvent === "stage") {
                  updateNode(parsed as StageEvent);
                } else if (currentEvent === "token") {
                  setStreamingAnswer((prev) => prev + (parsed.token ?? ""));
                } else if (currentEvent === "done") {
                  setAnswer(parsed.answer ?? "");
                  setStreamingAnswer("");
                  const f = parsed.follow_up_questions;
                  if (Array.isArray(f)) {
                    setFollowUps(
                      f
                        .map((q: unknown) => (typeof q === "string" ? q.trim() : ""))
                        .filter((q: string) => q.length > 0)
                        .slice(0, 3)
                    );
                  } else {
                    setFollowUps([]);
                  }
                } else if (currentEvent === "error") {
                  setError(parsed.message ?? "Unknown error");
                  setFollowUps([]);
                }
              } catch {
                // malformed JSON — skip
              }
              currentEvent = "";
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name !== "AbortError") {
          setError(err.message);
        }
      } finally {
        setStreaming(false);
      }
    },
    [reset, updateNode]
  );

  return { nodes, answer, streamingAnswer, streaming, error, submit, reset, followUps };
}
