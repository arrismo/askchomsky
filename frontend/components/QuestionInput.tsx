"use client";

import { useState, useRef, type KeyboardEvent } from "react";

interface QuestionInputProps {
  onSubmit: (question: string) => void;
  disabled?: boolean;
}

const EXAMPLES = [
  "What is Universal Grammar?",
  "How did Chomsky critique behaviorism?",
  "Compare Principles and Parameters with Minimalism.",
  "How did Chomsky's views on language acquisition evolve over time?",
  "What is Chomsky's theory of propaganda?",
];

export default function QuestionInput({ onSubmit, disabled }: QuestionInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleExample = (q: string) => {
    setValue(q);
    textareaRef.current?.focus();
  };

  return (
    <div className="flex flex-col gap-3 w-full">
      {/* Input box */}
      <div className="flex gap-2 items-end">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask something about Noam Chomsky's work…"
          rows={2}
          disabled={disabled}
          className="
            flex-1 resize-none rounded-xl border border-zinc-700 bg-zinc-900
            px-4 py-3 text-sm text-zinc-100 placeholder-zinc-500
            focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500
            disabled:opacity-50 disabled:cursor-not-allowed
            transition-colors
          "
        />
        <button
          onClick={handleSubmit}
          disabled={disabled}
          className="
            flex-shrink-0 h-12 px-5 rounded-xl text-sm font-semibold
            bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700
            text-white transition-colors
            disabled:opacity-40 disabled:cursor-not-allowed
            flex items-center gap-2
          "
        >
          {disabled ? (
            <>
              <span className="w-3 h-3 rounded-full border-2 border-white/40 border-t-white animate-spin" />
              Running
            </>
          ) : (
            "Ask →"
          )}
        </button>
      </div>

      {/* Example questions */}
      <div className="flex flex-wrap gap-2">
        <span className="text-[11px] text-zinc-600 self-center">Try:</span>
        {EXAMPLES.map((q) => (
          <button
            key={q}
            onClick={() => handleExample(q)}
            disabled={disabled}
            className="
              text-[11px] px-2.5 py-1 rounded-lg border border-zinc-700
              text-zinc-400 hover:text-zinc-200 hover:border-zinc-500
              bg-zinc-900 transition-colors disabled:opacity-40 disabled:cursor-not-allowed
            "
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
