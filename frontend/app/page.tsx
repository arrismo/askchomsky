"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import ReactMarkdown from "react-markdown";
import QuestionInput from "@/components/QuestionInput";
import { useQueryStream } from "@/hooks/useQueryStream";
import { PIPELINE_STAGES, type NodeState } from "@/types/pipeline";

const STAGE_DESCRIPTIONS: Record<string, string> = {
  intent:
    "Classifies the user message as a real corpus question vs. small talk or other intent, so the pipeline can short-circuit when appropriate.",
  rewrite:
    "Rewrites the question into a retrieval-optimized form while preserving meaning and key entities, to improve RAG recall.",
  rag_init:
    "Initializes the LightRAG store and loads the graph/vector index from disk for this session.",
  retrieval_1:
    "Runs the first retrieval pass over the Chomsky corpus using the selected mode and top-k settings.",
  retrieval_2:
    "If the first pass is weak, retries retrieval with more aggressive parameters (higher top-k / chunk_top_k).",
  retrieval_3:
    "Final retrieval retry with the most expansive settings to maximize recall when earlier passes are insufficient.",
  citations:
    "Summarizes which corpus files were used and ensures the answer will refer back to them with citation markers.",
  verification:
    "Uses a secondary verifier to check whether the answer is supported by the retrieved snippets and flags unsupported claims.",
  answer:
    "Streams the answer tokens, then post-processes with citations and claim verification before returning the final response.",
};

// React Flow must be loaded client-side only (uses browser APIs)
const FlowCanvas = dynamic(() => import("@/components/FlowCanvas"), { ssr: false });

export default function Home() {
  const { nodes, answer, streamingAnswer, streaming, error, submit, reset, followUps } = useQueryStream();
  const [mode, setMode] = useState<string>("hybrid");
  const [selectedStageId, setSelectedStageId] = useState<string>(PIPELINE_STAGES[0]);

  const handleSubmit = (question: string) => {
    reset();
    submit(question, mode);
  };

  const handleSelectStage = (id: string) => {
    if (PIPELINE_STAGES.includes(id as (typeof PIPELINE_STAGES)[number])) {
      setSelectedStageId(id);
    }
  };

  const selectedNode: NodeState | undefined = useMemo(
    () => nodes.find((n) => n.id === selectedStageId),
    [nodes, selectedStageId]
  );

  const hasStarted = nodes.some((n) => n.status !== "idle");

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-zinc-100">
      {/* ── Header ── */}
      <header className="flex-shrink-0 flex items-center justify-between px-6 py-4 border-b border-zinc-800">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🧠</span>
          <div>
            <h1 className="text-lg font-bold tracking-tight">AskChomsky</h1>
            <p className="text-[11px] text-zinc-500 leading-none mt-0.5">
              RAG pipeline visualizer · Noam Chomsky corpus
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-[11px] text-zinc-500">
            <span>Mode</span>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              disabled={streaming}
              className="bg-zinc-900 border border-zinc-700 rounded-md px-2 py-1 text-[11px] text-zinc-200 focus:outline-none focus:border-indigo-500 disabled:opacity-40"
            >
              <option value="naive">naive</option>
              <option value="local">local</option>
              <option value="global">global</option>
              <option value="hybrid">hybrid</option>
              <option value="mix">mix</option>
            </select>
          </div>
          {hasStarted && (
            <button
              onClick={reset}
              disabled={streaming}
              className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-500 transition-colors disabled:opacity-40"
            >
              Reset
            </button>
          )}
        </div>
      </header>

      {/* ── Body ── */}
      <div className="flex flex-1 min-h-0">
        {/* Left panel: flow diagram + stage detail */}
        <div className="flex-1 min-w-0 flex flex-col border-r border-zinc-900">
          <div className="flex-1 min-h-0 relative">
            {hasStarted ? (
              <FlowCanvas
                nodeStates={nodes}
                onSelectStage={handleSelectStage}
                selectedStageId={selectedStageId}
              />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
                <span className="text-5xl opacity-20">🔍</span>
                <p className="text-zinc-600 text-sm text-center max-w-xs leading-relaxed">
                  Ask a question below to watch the RAG pipeline come alive — each node lights up as the agent processes your query.
                </p>
              </div>
            )}
          </div>

          {/* Stage detail panel */}
          <div className="h-40 border-t border-zinc-900 bg-zinc-950/80 px-4 py-3 flex flex-col gap-1">
            <div className="flex items-center justify-between gap-2">
              <div className="flex flex-col">
                <span className="text-[11px] uppercase tracking-[0.12em] text-zinc-500">Stage details</span>
                <span className="text-sm font-semibold text-zinc-100 truncate">
                  {selectedNode?.label ?? "Select a stage"}
                </span>
              </div>
              <span className="text-[10px] px-2 py-0.5 rounded-full border border-zinc-700 text-zinc-400">
                {selectedNode?.status === "running"
                  ? "Running"
                  : selectedNode?.status === "done"
                  ? "Done"
                  : selectedNode?.status === "error"
                  ? "Error"
                  : "Idle"}
              </span>
            </div>

            <p className="text-[11px] text-zinc-400 leading-snug line-clamp-2">
              {selectedStageId && STAGE_DESCRIPTIONS[selectedStageId]
                ? STAGE_DESCRIPTIONS[selectedStageId]
                : "Click a node in the pipeline to see what it does."}
            </p>

            <div className="flex-1 min-h-0 mt-1 rounded-lg bg-zinc-950 border border-zinc-900 overflow-hidden">
              <div className="h-full w-full overflow-y-auto px-3 py-2">
                {selectedNode?.detail ? (
                  <pre className="text-[10px] leading-snug text-zinc-400 whitespace-pre-wrap font-mono">
                    {selectedNode.detail}
                  </pre>
                ) : (
                  <p className="text-[10px] text-zinc-600">
                    Live stage diagnostics will appear here as the pipeline runs.
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Right panel: answer */}
        <aside className="w-[380px] flex-shrink-0 border-l border-zinc-800 flex flex-col">
          <div className="px-5 py-3 border-b border-zinc-800">
            <h2 className="text-sm font-semibold text-zinc-300">Answer</h2>
          </div>
          <div className="flex-1 overflow-y-auto px-5 py-4">
            {/* Spinner only while pipeline stages run, before tokens start */}
            {streaming && !streamingAnswer && !answer && (
              <div className="flex items-center gap-2 text-zinc-500 text-sm">
                <span className="w-3 h-3 rounded-full border-2 border-zinc-600 border-t-blue-400 animate-spin" />
                Thinking…
              </div>
            )}

            {error && (
              <div className="rounded-xl bg-red-950/50 border border-red-800 p-4 text-sm text-red-300">
                <p className="font-semibold mb-1">Error</p>
                <p className="font-mono text-xs">{error}</p>
              </div>
            )}

            {/* Live token stream — plain text while streaming */}
            {streamingAnswer && !answer && !error && (
              <div className="text-sm text-zinc-300 leading-relaxed whitespace-pre-wrap">
                {streamingAnswer}
                <span className="inline-block w-0.5 h-3.5 bg-indigo-400 ml-0.5 align-middle animate-pulse" />
              </div>
            )}

            {/* Final answer with full markdown once streaming is done */}
            {answer && !error && (
              <div className="text-sm text-zinc-300 leading-relaxed [&>p]:mb-3 [&>h1]:text-base [&>h1]:font-bold [&>h1]:text-zinc-100 [&>h1]:mb-2 [&>h2]:text-sm [&>h2]:font-semibold [&>h2]:text-zinc-200 [&>h2]:mb-2 [&>h3]:text-sm [&>h3]:font-semibold [&>h3]:text-zinc-300 [&>h3]:mb-1 [&>ul]:list-disc [&>ul]:pl-4 [&>ul]:mb-3 [&>ul>li]:mb-1 [&>ol]:list-decimal [&>ol]:pl-4 [&>ol]:mb-3 [&>ol>li]:mb-1 [&>hr]:border-zinc-700 [&>hr]:my-4 [&>strong]:text-zinc-100 [&>blockquote]:border-l-2 [&>blockquote]:border-zinc-600 [&>blockquote]:pl-3 [&>blockquote]:text-zinc-400 [&>blockquote]:italic [&>code]:bg-zinc-800 [&>code]:px-1 [&>code]:rounded [&>code]:text-xs [&>code]:font-mono">
                <ReactMarkdown>{answer}</ReactMarkdown>
              </div>
            )}

            {/* Follow-up questions suggested by the backend */}
            {followUps.length > 0 && !error && (
              <div className="mt-6 border-t border-zinc-800 pt-4">
                <h3 className="text-xs font-semibold text-zinc-400 mb-2">
                  Follow-up questions
                </h3>
                <div className="flex flex-wrap gap-2">
                  {followUps.map((q) => (
                    <button
                      key={q}
                      type="button"
                      onClick={() => handleSubmit(q)}
                      disabled={streaming}
                      className="text-[11px] px-2.5 py-1 rounded-lg border border-zinc-700 text-zinc-300 hover:text-zinc-100 hover:border-zinc-500 bg-zinc-900 transition-colors disabled:opacity-40 disabled:cursor-not-allowed text-left"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {!streaming && !answer && !error && !streamingAnswer && (
              <p className="text-zinc-600 text-sm">
                The answer will appear here once the pipeline finishes.
              </p>
            )}
          </div>
        </aside>
      </div>

      {/* ── Footer input ── */}
      <footer className="flex-shrink-0 border-t border-zinc-800 px-6 py-4">
        <QuestionInput onSubmit={handleSubmit} disabled={streaming} />
      </footer>
    </div>
  );
}
