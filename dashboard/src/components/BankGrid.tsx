import type { Bank, BankMetricRow } from "../lib/types";
import BankCard from "./BankCard";

export default function BankGrid({
  banks, histories, onCardClick, onFault, onSwap,
}: {
  banks: Bank[];
  histories: Record<string, BankMetricRow[]>;
  onCardClick: (id: string) => void;
  onFault: (id: string) => void;
  onSwap: (id: string) => void;
}) {
  return (
    <div style={{ padding: "14px 18px" }}>
      <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 }}>
        Banks · click card for full local history · drop CSV to swap dataset
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
        {banks.map((b) => (
          <BankCard
            key={b.bank_id}
            bank={b}
            history={histories[b.bank_id] ?? []}
            onClick={() => onCardClick(b.bank_id)}
            onFault={() => onFault(b.bank_id)}
            onSwap={() => onSwap(b.bank_id)}
          />
        ))}
      </div>
    </div>
  );
}
