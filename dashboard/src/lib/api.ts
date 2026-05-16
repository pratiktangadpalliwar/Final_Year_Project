import type { Bank, BankMetricRow, GlobalMetricRow, RoundStatus, Fault } from "./types";

async function http<T>(method: string, url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    credentials: "include",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${method} ${url} → ${r.status}`);
  return r.json() as Promise<T>;
}

export const api = {
  login:  (password: string) => http<{ ok: boolean }>("POST", "/admin/login", { password }),
  logout: ()                  => http<{ ok: boolean }>("POST", "/admin/logout"),
  pause:  ()                  => http<{ paused: boolean }>("POST", "/admin/pause"),
  resume: ()                  => http<{ paused: boolean }>("POST", "/admin/resume"),
  reset:  ()                  => http<{ current_round: number }>("POST", "/admin/reset"),
  setFault: (bank_id: string, fault: Fault) =>
              http<{ bank_id: string; fault: Fault }>("POST", "/admin/fault", { bank_id, fault }),

  uploadDataset: async (bank_id: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`/admin/dataset/${bank_id}`, { method: "POST", body: fd, credentials: "include" });
    if (!r.ok) throw new Error(`upload → ${r.status}`);
    return r.json() as Promise<{ bank_id: string; dataset_version: number }>;
  },

  banks:        () => http<Bank[]>("GET", "/banks"),
  bankHistory:  (bank_id: string, n = 50) =>
                  http<BankMetricRow[]>("GET", `/banks/${bank_id}/history?n=${n}`),
  metrics:      (n = 50) =>
                  http<{ history: GlobalMetricRow[]; cumulative_eps_global: number; current_round: number }>(
                    "GET", `/metrics?n=${n}`,
                  ),
  roundStatus:  () => http<RoundStatus>("GET", "/round/status"),
};
