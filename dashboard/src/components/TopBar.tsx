import type { RoundStatus } from "../lib/types";
import { api } from "../lib/api";
import { performLogout } from "../lib/auth-context";
import { useNavigate } from "react-router-dom";

export default function TopBar({ status, banks, eps, onChange }: {
  status: RoundStatus; banks: number; eps: number; onChange: () => void;
}) {
  const nav = useNavigate();
  const action = (fn: () => Promise<unknown>) => async () => { await fn(); onChange(); };

  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "12px 18px", borderBottom: "1px solid #30363d", background: "#161b22" }}>
      <div>
        <span style={{ fontWeight: 700, fontSize: 15 }}>FL Demo</span>
        <span style={{ opacity: 0.6, marginLeft: 10, fontSize: 12 }}>
          round&nbsp;<b style={{ color: "#58a6ff" }}>{status.round}</b> ·
          state&nbsp;<b style={{ color: status.state === "stalled" ? "#f85149" : "#3fb950" }}>{status.state}</b> ·
          banks&nbsp;<b>{banks}</b> · ε&nbsp;<b>{eps.toFixed(2)}</b>
        </span>
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button style={ctrl} onClick={action(api.resume)}>▶ Resume</button>
        <button style={ctrl} onClick={action(api.pause)}>⏸ Pause</button>
        <button style={ctrl} onClick={action(api.reset)}>↺ Reset</button>
        <button style={{ ...ctrl, borderColor: "#f85149", color: "#f85149" }}
                onClick={async () => { await performLogout(); nav("/login"); }}>Logout</button>
      </div>
    </div>
  );
}

const ctrl: React.CSSProperties = {
  background: "#21262d", border: "1px solid #30363d", color: "#e6edf3",
  padding: "5px 12px", borderRadius: 6, fontSize: 12, cursor: "pointer",
};
