export type Status = "auto" | "asked" | "needs_review";

export interface Evidence {
  frame: string;
  t: number;
  sharpness: number;
  head_speed: number;
}

export interface Alternative {
  san: string;
  uci: string | null;
  p: number;
}

export interface Ply {
  ply: number;
  t: number;
  san: string | null;
  uci: string | null;
  status: Status;
  confidence: number;
  note?: string;
  first_guess?: { san: string; uci: string; p: number } | null;
  alternatives: Alternative[];
  question: { text: string; options: string[]; answer: string | null } | null;
  evidence: Evidence[];
  degraded: boolean;
}

export interface RecordFile {
  schema: string;
  synthetic: boolean;
  meta: Record<string, unknown>;
  stats: { total_frames: number; decoded_frames: number; selected_frames: number; degraded_events: string[] };
  plies: Ply[];
  pgn: string;
}

export interface ApprovedPly {
  ply: number;
  uci: string;
  san: string;
  source: "auto" | "asked" | "review_kept" | "review_changed" | "corrected";
}

export interface ApprovedRecord {
  schema: "boardscribe/approved@1";
  synthetic: boolean;
  plies: ApprovedPly[];
  review: { seconds: number; corrections: number; flagged_at_open: number };
}
