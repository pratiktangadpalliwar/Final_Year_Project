import type { GlobalMetricRow } from "../lib/types";
import Sparkline from "./Sparkline";

const SPECS: { key: keyof GlobalMetricRow; label: string; color: string }[] = [
  { key: "auc",        label: "AUC-ROC",   color: "#58a6ff" },
  { key: "f1",         label: "F1",        color: "#a371f7" },
  { key: "precision",  label: "Precision", color: "#3fb950" },
  { key: "recall",     label: "Recall",    color: "#d29922" },
  { key: "val_loss",   label: "Val loss",  color: "#f85149" },
];

export default function GlobalMetrics({ history }: { history: GlobalMetricRow[] }) {
  return (
    <div style={{ padding: "14px 18px", borderBottom: "1px solid #30363d" }}>
      <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 }}>
        Global model — last {history.length} rounds
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 8 }}>
        {SPECS.map(({ key, label, color }) => {
          const series = history.map((r) => r[key]).filter((v): v is number => typeof v === "number");
          const current = series.length ? series[series.length - 1].toFixed(3) : "—";
          return (
            <div key={String(key)} style={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 6, padding: 8 }}>
              <div style={{ fontSize: 10, opacity: 0.7, display: "flex", justifyContent: "space-between" }}>
                <span>{label}</span>
                <b style={{ color }}>{current}</b>
              </div>
              <Sparkline values={series} color={color} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
