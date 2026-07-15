export type Theater = "forge" | "solve" | "verify" | "atlas" | "autopilot" | "doctor";
export type Verdict = "pass" | "warn" | "fail" | "idle";

export interface GeometrySpec {
  kind: "box" | "cylinder" | "sphere" | "extrude";
  width?: number;
  height?: number;
  depth?: number;
  radius?: number;
  profile?: [number, number][];
  sculpt_offset?: number;
  features?: Array<Record<string, unknown>>;
}

export interface DoctorProbe {
  backend: string;
  available: boolean;
  reason: string;
  details: Record<string, unknown>;
}

export interface BootstrapData {
  appRoot: string;
  repoRoot: string;
  doctor: {
    native_ready: boolean;
    ready_backends: string[];
    missing_backends: string[];
    probes: Record<string, DoctorProbe>;
    runtime: Record<string, unknown>;
  };
  stats: {
    runs: number;
    verified: number;
    promoted: number;
    momentum: number;
    modelVersion: string;
  };
  recentRuns: RunSummary[];
}

export interface RunSummary {
  id: string;
  name: string;
  recordedAt: string;
  verified: boolean;
  status: string;
  solver: string;
  solverMode: string;
  objective?: number;
  volume?: number;
  findings: string[];
  tags: string[];
  artifacts: string[];
}

export interface PipelineResult {
  runId: string;
  ok: boolean;
  run: {
    status: string;
    manifest: Record<string, unknown>;
    provenance: Record<string, unknown>;
  };
  solver_result: {
    status: string;
    objective?: number;
    residual?: number;
    iterations?: number;
    metadata: Record<string, unknown>;
    artifacts: string[];
  };
  verification: {
    name: string;
    passed: boolean;
    findings: string[];
    metrics: Record<string, unknown>;
    backend: string;
  };
  artifacts: string[];
  report_text: string;
  geometry: GeometrySpec;
  ghosts: Array<{ iteration: number; scale: number; geometry: GeometrySpec }>;
  metrics: Record<string, number | string | boolean | null>;
  momentumEarned: number;
}

export interface BridgeEvent {
  event: string;
  payload: {
    stage?: string;
    progress?: number;
    message?: string;
    ok?: boolean;
    level?: string;
  };
}

declare global {
  interface Window {
    lattice?: {
      request: <T>(method: string, params?: Record<string, unknown>) => Promise<T>;
      chooseDirectory: () => Promise<string | null>;
      reveal: (path: string) => Promise<void>;
      open: (path: string) => Promise<void>;
      onEvent: (callback: (event: BridgeEvent) => void) => () => void;
      platform: string;
    };
  }
}
