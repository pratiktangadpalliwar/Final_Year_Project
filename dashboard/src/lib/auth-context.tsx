import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "./api";

type AuthState = { loggedIn: boolean; loading: boolean };
const AuthCtx = createContext<{ state: AuthState; refresh: () => Promise<void> }>({
  state: { loggedIn: false, loading: true },
  refresh: async () => {},
});

// Plan 2: trust localStorage flag set on successful login. The cookie is
// HttpOnly so JS can't read it; cookie validity is enforced server-side on
// every gated request. If the flag drifts (cookie expired) the next admin
// call will 401 and the user is re-prompted.
export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({ loggedIn: false, loading: true });

  const refresh = async () => {
    const v = localStorage.getItem("fl_logged_in") === "1";
    setState({ loggedIn: v, loading: false });
  };

  useEffect(() => { refresh(); }, []);

  return <AuthCtx.Provider value={{ state, refresh }}>{children}</AuthCtx.Provider>;
}

export const useAuth = () => useContext(AuthCtx);

export async function performLogin(password: string) {
  await api.login(password);
  localStorage.setItem("fl_logged_in", "1");
}

export async function performLogout() {
  try { await api.logout(); } catch { /* server may be down; clear anyway */ }
  localStorage.removeItem("fl_logged_in");
}
