import { useRef, useState } from "react";
import { api } from "../lib/api";

export default function DatasetUpload({
  bankId, onClose,
}: { bankId: string; onClose: (success: boolean) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const upload = async () => {
    const f = inputRef.current?.files?.[0];
    if (!f) { setErr("pick a file first"); return; }
    setBusy(true); setErr(null);
    try {
      await api.uploadDataset(bankId, f);
      onClose(true);
    } catch (e: unknown) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={overlay} onClick={() => onClose(false)}>
      <div style={panel} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ marginTop: 0 }}>Swap dataset for {bankId}</h3>
        <input ref={inputRef} type="file" accept=".csv" />
        <div style={{ marginTop: 14, display: "flex", gap: 8 }}>
          <button onClick={upload} disabled={busy} style={primary}>
            {busy ? "uploading…" : "Upload"}
          </button>
          <button onClick={() => onClose(false)} style={secondary}>Cancel</button>
        </div>
        {err && <p style={{ color: "#f85149", marginTop: 12 }}>{err}</p>}
      </div>
    </div>
  );
}

const overlay: React.CSSProperties = { position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)",
  display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100 };
const panel: React.CSSProperties = { background: "#161b22", border: "1px solid #30363d",
  borderRadius: 8, padding: 16, minWidth: 340 };
const primary: React.CSSProperties = { background: "#1f6feb", color: "#fff",
  border: 0, borderRadius: 6, padding: "6px 14px", cursor: "pointer" };
const secondary: React.CSSProperties = { background: "#21262d", color: "#e6edf3",
  border: "1px solid #30363d", borderRadius: 6, padding: "6px 14px", cursor: "pointer" };
