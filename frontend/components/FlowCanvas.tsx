"use client";

import { useEffect, useMemo } from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  useNodesState,
  useEdgesState,
  MarkerType,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import PipelineNode from "./PipelineNode";
import type { NodeState } from "@/types/pipeline";

interface PipelineNodeDataShape extends Record<string, unknown> {
  label: string;
  status: string;
  detail: string;
  icon: string;
  selected?: boolean;
}

// ── Icon map ────────────────────────────────────────────────────────────────
const ICONS: Record<string, string> = {
  intent: "🧭",
  rewrite: "✏️",
  cache: "⚡",
  rag_init: "🗄️",
  retrieval_1: "🔍",
  retrieval_2: "🔄",
  retrieval_3: "🔄",
  citations: "📎",
  verification: "✅",
  answer: "💬",
};

// ── Fixed layout positions (x, y) ──────────────────────────────────────────
// Single vertical column; retrieval retries branch slightly to the right
const POSITIONS: Record<string, { x: number; y: number }> = {
  intent:       { x: 0, y: 0 },
  rewrite:      { x: 0, y: 160 },
  cache:        { x: 300, y: 160 },
  rag_init:     { x: 0, y: 320 },
  retrieval_1:  { x: 0, y: 480 },
  retrieval_2:  { x: 300, y: 480 },
  retrieval_3:  { x: 600, y: 480 },
  citations:    { x: 0, y: 640 },
  verification: { x: 0, y: 800 },
  answer:       { x: 0, y: 960 },
};

// ── Edge definitions ────────────────────────────────────────────────────────
const STATIC_EDGES: Edge[] = [
  { id: "e-intent-rewrite",       source: "intent",      target: "rewrite" },
  { id: "e-rewrite-cache",        source: "rewrite",     target: "cache" },
  { id: "e-cache-rag",            source: "cache",       target: "rag_init" },
  { id: "e-rag-r1",               source: "rag_init",    target: "retrieval_1" },
  { id: "e-r1-r2",                source: "retrieval_1", target: "retrieval_2" },
  { id: "e-r2-r3",                source: "retrieval_2", target: "retrieval_3" },
  { id: "e-r1-citations",         source: "retrieval_1", target: "citations" },
  { id: "e-r2-citations",         source: "retrieval_2", target: "citations" },
  { id: "e-r3-citations",         source: "retrieval_3", target: "citations" },
  { id: "e-citations-verify",     source: "citations",   target: "verification" },
  { id: "e-verify-answer",        source: "verification",target: "answer" },
].map((e) => ({
  ...e,
  animated: false,
  style: { stroke: "#3f3f46", strokeWidth: 2 },
  markerEnd: { type: MarkerType.ArrowClosed, color: "#3f3f46" },
}));

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const nodeTypes: Record<string, any> = { pipeline: PipelineNode };

interface FlowCanvasProps {
  nodeStates: NodeState[];
  onSelectStage?: (id: string) => void;
  selectedStageId?: string | null;
}

export default function FlowCanvas({ nodeStates, onSelectStage, selectedStageId }: FlowCanvasProps) {
  const initialNodes: Node<PipelineNodeDataShape>[] = useMemo(
    () =>
      nodeStates.map((ns) => ({
        id: ns.id,
        type: "pipeline" as const,
        position: POSITIONS[ns.id] ?? { x: 0, y: 0 },
        data: {
          label: ns.label,
          status: ns.status,
          detail: ns.detail,
          icon: ICONS[ns.id] ?? "⚙️",
          selected: ns.id === selectedStageId,
        },
        draggable: true,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [] // only for initial render — we sync via useEffect below
  );

  const [nodes, setNodes, onNodesChange] = useNodesState<Node<PipelineNodeDataShape>>(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(STATIC_EDGES);

  // Sync external nodeStates into React Flow node data
  useEffect(() => {
    setNodes((prev) =>
      prev.map((n) => {
        const updated = nodeStates.find((ns) => ns.id === n.id);
        if (!updated) return n;
        return {
          ...n,
          data: {
            ...n.data,
            label: updated.label,
            status: updated.status,
            detail: updated.detail,
            selected: selectedStageId ? updated.id === selectedStageId : (n.data as PipelineNodeDataShape).selected,
          },
        };
      })
    );
  }, [nodeStates, selectedStageId, setNodes]);

  // Animate edges that connect active/done nodes
  const animatedEdges = useMemo(() => {
    const doneOrRunning = new Set(
      nodeStates.filter((n) => n.status === "done" || n.status === "running").map((n) => n.id)
    );
    return edges.map((e) => {
      const active = doneOrRunning.has(e.source) && doneOrRunning.has(e.target);
      return {
        ...e,
        animated: active,
        style: {
          stroke: active ? "#6366f1" : "#3f3f46",
          strokeWidth: active ? 2.5 : 2,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: active ? "#6366f1" : "#3f3f46",
        },
      };
    });
  }, [edges, nodeStates]);

  return (
    <div className="w-full h-full">
      <ReactFlow
        nodes={nodes}
        edges={animatedEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => {
          if (onSelectStage) {
            onSelectStage(node.id);
          }
        }}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.2}
        maxZoom={2}
        colorMode="dark"
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#27272a" />
        <Controls className="!bg-zinc-900 !border-zinc-700 !text-zinc-300" />
      </ReactFlow>
    </div>
  );
}
