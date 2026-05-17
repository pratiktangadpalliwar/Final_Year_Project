import type { WsEvent } from "../lib/types";

const COLOR: Record<string, string> = {
  info: "#58a6ff", warn: "#d29922", error: "#f85149", ok: "#3fb950",
};

function describe(e: WsEvent): { color: string; text: string } {
  switch (e.type) {
    case "round_started":   return { color: "info",  text: `round ${e.round} started — quorum ${e.quorum_size} needed` };
    case "round_completed": return { color: "ok",    text: `round ${e.round} published via ${e.method}` };
    case "round_stalled":   return { color: "warn",  text: `round ${e.round} stalled (${e.received} received${e.reason ? `, ${e.reason}` : ""})` };
    case "round_rolled_back":
      return {
        color: "warn",
        text: e.reason === "eval_failed"
          ? `round ${e.round} ROLLED BACK (eval failed)`
          : `round ${e.round} ROLLED BACK (auc ${(e.candidate_auc ?? 0).toFixed(3)} < ${(e.prev_auc ?? 0).toFixed(3)})`,
      };
    case "bank_update":     return { color: "ok",    text: `${e.bank_id} round ${e.round} auc=${(e.metrics.val_auc ?? 0).toFixed(2)}` };
    case "event":           return { color: e.level, text: e.msg };
  }
}

export default function EventLog({ events }: { events: WsEvent[] }) {
  return (
    <div style={{ padding: "14px 18px", borderTop: "1px solid #30363d" }}>
      <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 }}>
        Event log (live, WebSocket)
      </div>
      <div style={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 6,
                    padding: 10, fontFamily: "monospace", fontSize: 11, lineHeight: 1.7,
                    maxHeight: 200, overflow: "auto" }}>
        {events.slice(-50).map((e, i) => {
          const { color, text } = describe(e);
          const ts = new Date().toLocaleTimeString();
          return (
            <div key={i}>
              <span style={{ color: COLOR[color] }}>{ts}</span> {text}
            </div>
          );
        })}
      </div>
    </div>
  );
}
