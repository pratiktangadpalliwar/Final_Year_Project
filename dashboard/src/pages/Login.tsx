import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { performLogin, useAuth } from "../lib/auth-context";

export default function Login() {
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const nav = useNavigate();
  const { refresh } = useAuth();

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    try {
      await performLogin(password);
      await refresh();
      nav("/");
    } catch {
      setErr("invalid password");
    }
  };

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh" }}>
      <form onSubmit={submit} style={{ background: "#161b22", padding: 24, borderRadius: 8, minWidth: 320 }}>
        <h2 style={{ marginTop: 0 }}>FL Demo</h2>
        <input
          type="password"
          autoFocus
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="admin password"
          style={{ width: "100%", padding: 10, fontSize: 14, marginBottom: 12,
                   background: "#0e1117", color: "#e6edf3", border: "1px solid #30363d", borderRadius: 6 }}
        />
        <button type="submit" style={{ width: "100%", padding: 10, background: "#1f6feb", color: "#fff",
                                       border: 0, borderRadius: 6, fontSize: 14, cursor: "pointer" }}>
          Sign in
        </button>
        {err && <p style={{ color: "#f85149", marginTop: 12 }}>{err}</p>}
      </form>
    </div>
  );
}
