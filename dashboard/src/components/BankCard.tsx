import type { Bank, BankMetricRow, Fault } from "../lib/types";
import Sparkline from "./Sparkline";

const STATUS_COLOR: Record<Fault, string> = {
  none: "#3fb950", crash: "#f85149", straggle: "#d29922",
  byzantine: "#d29922", partition: "#f85149",
};

export default function BankCard({
  bank, history, onClick, onFault, onSwap,
}: {
  bank: Bank; history: BankMetricRow[];
  onClick: () => void;
  onFault: () => void;
  onSwap: () => void;
}) {
  const series = history.map((r) => r.val_auc).filter((v): v is number => v != null);
  const last = history.at(-1) ?? ({} as BankMetricRow);
  const border = bank.fault === "none" ? "#30363d" : STATUS_COLOR[bank.fault];
  return (
    <div
      onClick={onClick}
      style={{ background: "#161b22", border: `1px solid ${border}`, borderRadius: 8,
               padding: 10, cursor: "pointer" }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start" }}>
        <div>
          <div style={{ fontWeight: 600, fontSize: 13 }}>{bank.bank_id}</div>
          <div style={{ fontSize: 10, opacity: 0.6 }}>
            trust {bank.trust.toFixed(2)} · ds v{bank.dataset_version} · n={bank.n_samples}
          </div>
        </div>
        <span style={{ background: STATUS_COLOR[bank.fault] + "33", color: STATUS_COLOR[bank.fault],
                       padding: "2px 7px", borderRadius: 10, fontSize: 10 }}>
          {bank.fault === "none" ? "idle" : bank.fault}
        </span>
      </div>
      <Sparkline values={series} color="#58a6ff" height={30} />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 4,
                    fontFamily: "monospace", fontSize: 9.5, marginTop: 6 }}>
        <div><span style={{ opacity: 0.6 }}>loss</span> <b>{last.val_loss?.toFixed(3) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>auc</span>  <b>{last.val_auc?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>f1</span>   <b>{last.val_f1?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>prec</span> <b>{last.val_precision?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>rec</span>  <b>{last.val_recall?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>acc</span>  <b>{last.val_accuracy?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>‖w‖</span>  <b>{last.weight_norm?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>σ</span>    <b>{last.dp_sigma?.toFixed(2) ?? "—"}</b></div>
        <div><span style={{ opacity: 0.6 }}>r#</span>   <b>{last.round ?? "—"}</b></div>
      </div>
      <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
        <button onClick={(e) => { e.stopPropagation(); onFault(); }} style={btn(border)}>⚠ fault</button>
        <button onClick={(e) => { e.stopPropagation(); onSwap(); }} style={btn("#30363d")}>📂 swap</button>
      </div>
    </div>
  );
}

const btn = (border: string): React.CSSProperties => ({
  flex: 1, background: "#21262d", border: `1px solid ${border}`,
  color: "#e6edf3", padding: 3, borderRadius: 4, fontSize: 10, cursor: "pointer",
});
