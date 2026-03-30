export default function Icon() {
  // Simple circular icon with "AC" monogram for AskChomsky
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 64 64"
    >
      <rect width="64" height="64" rx="16" fill="#020617" />
      <circle cx="20" cy="32" r="10" fill="#4f46e5" />
      <circle cx="44" cy="32" r="10" fill="#22c55e" />
      <text
        x="32"
        y="37"
        textAnchor="middle"
        fontFamily="system-ui, -apple-system, BlinkMacSystemFont, sans-serif"
        fontSize="14"
        fontWeight="700"
        fill="#e5e7eb"
      >
        AC
      </text>
    </svg>
  );
}
