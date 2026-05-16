import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { liveSocket } from "../lib/ws";
import type { Bank, BankMetricRow, GlobalMetricRow, RoundStatus, WsEvent } from "../lib/types";
import TopBar from "../components/TopBar";
import GlobalMetrics from "../components/GlobalMetrics";
import BankGrid from "../components/BankGrid";
import EventLog from "../components/EventLog";
import BankDrillIn from "../components/BankDrillIn";
import DatasetUpload from "../components/DatasetUpload";
import FaultPanel from "../components/FaultPanel";

const POLL_MS = 2000;

export default function Dashboard() {
  const [banks, setBanks] = useState<Bank[]>([]);
  const [status, setStatus] = useState<RoundStatus>({ round: 0, state: "idle", paused: false, active_banks: 0, quorum_size: 0 });
  const [globalHistory, setGlobalHistory] = useState<GlobalMetricRow[]>([]);
  const [bankHistories, setBankHistories] = useState<Record<string, BankMetricRow[]>>({});
  const [eps, setEps] = useState(0);
  const [events, setEvents] = useState<WsEvent[]>([]);
  const [drillIn, setDrillIn] = useState<string | null>(null);
  const [uploadFor, setUploadFor] = useState<string | null>(null);
  const [faultFor, setFaultFor] = useState<string | null>(null);

  const refresh = async () => {
    const [b, s, m] = await Promise.all([api.banks(), api.roundStatus(), api.metrics(50)]);
    setBanks(b); setStatus(s); setGlobalHistory(m.history); setEps(m.cumulative_eps_global);
    const histories: Record<string, BankMetricRow[]> = {};
    await Promise.all(b.map(async (bk) => {
      histories[bk.bank_id] = await api.bankHistory(bk.bank_id, 50);
    }));
    setBankHistories(histories);
  };

  useEffect(() => {
    refresh().catch(console.warn);
    const id = setInterval(() => refresh().catch(console.warn), POLL_MS);
    liveSocket.connect();
    const unsub = liveSocket.subscribe((e) => {
      setEvents((prev) => [...prev, e].slice(-100));
      if (e.type === "round_completed" || e.type === "round_stalled" || e.type === "round_rolled_back") {
        refresh().catch(console.warn);
      }
    });
    return () => { clearInterval(id); unsub(); };
  }, []);

  const currentFault = useMemo(
    () => banks.find((b) => b.bank_id === faultFor)?.fault ?? "none",
    [banks, faultFor],
  );

  return (
    <div>
      <TopBar status={status} banks={banks.length} eps={eps} onChange={refresh} />
      <GlobalMetrics history={globalHistory} />
      <BankGrid
        banks={banks}
        histories={bankHistories}
        onCardClick={setDrillIn}
        onFault={setFaultFor}
        onSwap={setUploadFor}
      />
      <EventLog events={events} />
      {drillIn && (
        <BankDrillIn bankId={drillIn} local={bankHistories[drillIn] ?? []} global_={globalHistory}
                     onClose={() => setDrillIn(null)} />
      )}
      {uploadFor && (
        <DatasetUpload bankId={uploadFor} onClose={(ok) => { setUploadFor(null); if (ok) refresh(); }} />
      )}
      {faultFor && (
        <FaultPanel bankId={faultFor} currentFault={currentFault}
                    onClose={() => { setFaultFor(null); refresh(); }} />
      )}
    </div>
  );
}
