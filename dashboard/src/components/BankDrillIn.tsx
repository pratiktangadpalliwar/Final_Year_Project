import type { BankMetricRow, GlobalMetricRow } from "../lib/types";
import Sparkline from "./Sparkline";

export default function BankDrillIn({
  bankId, local, global_, onClose,
}: {
  bankId: string; local: BankMetricRow[]; global_: GlobalMetricRow[]; onClose: () => void;
}) {
  const last = local.at(-1) ?? ({} as BankMetricRow);
  return (
    <div style={overlay} onClick={onClose}>
      <div style={panel} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
          <div>
            <div style={{ fontWeight: 600, fontSize: 14 }}>{bankId} — local history</div>
            <div style={{ fontSize: 10, opacity: 0.6 }}>{local.length} rounds</div>
          </div>
          <button onClick={onClose} style={closeBtn}>close ✕</button>
        </div>
        <div style={{ fontSize: 10, opacity: 0.7, margin: "6px 0", textTransform: "uppercase" }}>
          LOCAL val_auc vs GLOBAL auc
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <Sparkline values={local.map((r) => r.val_auc ?? 0)} color="#d29922" height={80} />
          <Sparkline values={global_.map((r) => r.auc ?? 0)} color="#3fb950" height={80} />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8, marginTop: 12,
                      fontFamily: "monospace", fontSize: 10 }}>
          {(Object.entries(last) as [keyof BankMetricRow, number | undefined][]).map(([k, v]) => (
            <div key={k}>
              <div style={{ opacity: 0.6 }}>{k}</div>
              <b>{v != null ? (typeof v === "number" ? v.toFixed(3) : v) : "—"}</b>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const overlay: React.CSSProperties = {
  position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)",
  display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100,
};
const panel: React.CSSProperties = {
  background: "#161b22", border: "1px solid #30363d", borderRadius: 8,
  padding: 16, maxWidth: 800, width: "90%", maxHeight: "80vh", overflow: "auto",
};
const closeBtn: React.CSSProperties = {
  background: "#21262d", border: "1px solid #30363d", color: "#e6edf3",
  padding: "4px 10px", borderRadius: 4, fontSize: 11, cursor: "pointer",
};
