import { useNavigate } from "react-router-dom";
import { performLogout } from "../lib/auth-context";

export default function Dashboard() {
  const nav = useNavigate();
  const logout = async () => {
    await performLogout();
    nav("/login");
  };
  return (
    <div style={{ padding: 24 }}>
      <h1>FL Demo (placeholder)</h1>
      <button onClick={logout}>Logout</button>
    </div>
  );
}
