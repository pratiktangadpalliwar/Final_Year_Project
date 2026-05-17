type Props = { values: number[]; color: string; height?: number };

export default function Sparkline({ values, color, height = 50 }: Props) {
  if (values.length < 2) {
    return <svg width="100%" height={height}><text x="4" y="14" fill="#7d8590" fontSize="10">no data</text></svg>;
  }
  const W = 200, H = height;
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const dx = W / (values.length - 1);
  const points = values.map((v, i) => `${i * dx},${H - ((v - min) / span) * (H - 4) - 2}`).join(" L ");
  const path = `M ${points}`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: "100%", height }}>
      <path d={path} stroke={color} strokeWidth={1.5} fill="none" />
      <path d={`${path} L ${W},${H} L 0,${H} Z`} fill={color} fillOpacity={0.15} />
    </svg>
  );
}
