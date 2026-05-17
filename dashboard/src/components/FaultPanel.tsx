import { api } from "../lib/api";
import type { Fault } from "../lib/types";

const FAULTS: Fault[] = ["none", "crash", "straggle", "byzantine", "partition"];

export default function FaultPanel({
  bankId, currentFault, onClose,
}: { bankId: string; currentFault: Fault; onClose: () => void }) {
  const set = async (f: Fault) => { await api.setFault(bankId, f); onClose(); };
  return (
    <div style={overlay} onClick={onClose}>
      <div style={panel} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ marginTop: 0 }}>Fault for {bankId}</h3>
        <p style={{ fontSize: 11, opacity: 0.7 }}>currently: {currentFault}</p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {FAULTS.map((f) => (
            <button key={f} onClick={() => set(f)}
                    style={{ ...btn, ...(currentFault === f ? { borderColor: "#1f6feb" } : {}) }}>
              {f}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

const overlay: React.CSSProperties = { position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)",
  display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100 };
const panel: React.CSSProperties = { background: "#161b22", border: "1px solid #30363d",
  borderRadius: 8, padding: 16, minWidth: 340 };
const btn: React.CSSProperties = { background: "#21262d", color: "#e6edf3",
  border: "1px solid #30363d", borderRadius: 6, padding: "6px 14px", cursor: "pointer", fontSize: 12 };
