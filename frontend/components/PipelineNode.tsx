"use client";

import { memo } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import type { StageStatus } from "@/types/pipeline";

export type PipelineNodeData = Node<{
  label: string;
  status: StageStatus;
  detail: string;
  icon: string;
  selected?: boolean;
}>;

const statusStyles: Record<StageStatus, string> = {
  idle: "border-zinc-700 bg-zinc-900 text-zinc-500",
  running: "border-blue-500 bg-zinc-900 text-zinc-100 shadow-[0_0_16px_2px_rgba(59,130,246,0.4)]",
  done: "border-emerald-500 bg-zinc-900 text-zinc-100 shadow-[0_0_12px_2px_rgba(16,185,129,0.3)]",
  error: "border-red-500 bg-zinc-900 text-red-300 shadow-[0_0_12px_2px_rgba(239,68,68,0.4)]",
};

const statusDot: Record<StageStatus, string> = {
  idle: "bg-zinc-600",
  running: "bg-blue-400 animate-pulse",
  done: "bg-emerald-400",
  error: "bg-red-400",
};

const statusLabel: Record<StageStatus, string> = {
  idle: "Waiting",
  running: "Running…",
  done: "Done",
  error: "Error",
};

function PipelineNode({ data }: NodeProps<PipelineNodeData>) {
  const { label, status, detail, icon, selected } = data as {
    label: string;
    status: StageStatus;
    detail: string;
    icon: string;
    selected?: boolean;
  };

  return (
    <div
      className={`
        relative rounded-xl border-2 p-4 w-64 min-h-[90px] transition-all duration-300 cursor-pointer
        ${statusStyles[status]}
        ${selected ? "ring-2 ring-indigo-400 ring-offset-2 ring-offset-zinc-950" : ""}
      `}
    >
      {/* Top handle */}
      <Handle type="target" position={Position.Top} className="!bg-zinc-600 !border-zinc-500 !w-2 !h-2" />

      {/* Header row */}
      <div className="flex items-center gap-2 mb-2">
        <span className="text-lg leading-none">{icon}</span>
        <span className="font-semibold text-sm flex-1 leading-tight">{label}</span>
        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${statusDot[status]}`} />
      </div>

      {/* Status badge */}
      <div className="mb-2">
        <span
          className={`text-[10px] font-mono uppercase tracking-widest px-1.5 py-0.5 rounded ${
            status === "running"
              ? "bg-blue-900/60 text-blue-300"
              : status === "done"
              ? "bg-emerald-900/60 text-emerald-300"
              : status === "error"
              ? "bg-red-900/60 text-red-300"
              : "bg-zinc-800 text-zinc-500"
          }`}
        >
          {statusLabel[status]}
        </span>
      </div>

      {/* Detail text */}
      {detail && (
        <div className="mt-2 text-[11px] leading-relaxed text-zinc-400 font-mono whitespace-pre-wrap max-h-40 overflow-y-auto border-t border-zinc-800 pt-2">
          {detail}
        </div>
      )}

      {/* Bottom handle */}
      <Handle type="source" position={Position.Bottom} className="!bg-zinc-600 !border-zinc-500 !w-2 !h-2" />
    </div>
  );
}

export default memo(PipelineNode);
