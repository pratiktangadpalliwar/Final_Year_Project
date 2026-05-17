export type Fault = "none" | "crash" | "straggle" | "byzantine" | "partition";
export type RoundState = "idle" | "collecting" | "aggregating" | "stalled";

export interface Bank {
  bank_id: string;
  bank_name: string;
  n_samples: number;
  trust: number;
  suspended: boolean;
  dataset_version: number;
  fault: Fault;
  cumulative_eps: number;
}

export interface RoundStatus {
  round: number;
  state: RoundState;
  paused: boolean;
  active_banks: number;
  quorum_size: number;
}

export interface BankMetricRow {
  round: number;
  train_loss?: number;
  val_loss?: number;
  val_auc?: number;
  val_f1?: number;
  val_precision?: number;
  val_recall?: number;
  val_accuracy?: number;
  weight_norm?: number;
  dp_sigma?: number;
}

export interface GlobalMetricRow {
  round: number;
  method?: string;
  n_participants?: number;
  n_suspicious?: number;
  auc?: number;
  f1?: number;
  precision?: number;
  recall?: number;
  accuracy?: number;
  val_loss?: number;
}

export type WsEvent =
  | { type: "round_started"; round: number; quorum_size: number }
  | { type: "round_completed"; round: number; method: string }
  | { type: "round_stalled"; round: number; received: number; reason?: string }
  | { type: "round_rolled_back"; round: number; prev_auc?: number; candidate_auc?: number; reason?: string }
  | { type: "bank_update"; bank_id: string; round: number; metrics: Record<string, number> }
  | { type: "event"; level: "info" | "warn" | "error"; msg: string };
