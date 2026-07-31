export type Theater = "forge" | "solve" | "verify" | "atlas" | "autopilot" | "materials" | "doctor";
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

export interface SpaceMaterialProps {
  material_id: string;
  name: string;
  category: string;
  density_kg_m3: number;
  youngs_modulus_gpa: number;
  yield_mpa: number | null;
  ultimate_mpa: number | null;
  max_service_temp_k: number;
  cte_1e6_k?: number | null;
  thermal_conductivity_w_mk?: number | null;
  poisson_ratio?: number;
  shear_modulus_gpa?: number | null;
  allowable_stress_mpa?: number | null;
  property_source?: string;
  notes?: string;
}

export interface MaterialEvalSuite {
  framework: string;
  cases: number;
  passed: number;
  failed: number;
  pass_rate: number;
  materials_catalog_size: number;
  report_path?: string;
  reports: Array<{
    case_id: string;
    material_id: string;
    material_name: string;
    passed: boolean;
    mass_kg?: number | null;
    allowable_mpa?: number | null;
    checks: Array<{ name: string; passed: boolean; severity: string; message: string }>;
  }>;
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
  materials?: {
    catalog_path: string;
    materials: number;
    eval_report: string;
    suite_passed: number;
    suite_failed: number;
    pass_rate: number;
  };
  stats: {
    runs: number;
    verified: number;
    promoted: number;
    momentum: number;
    modelVersion: string;
    materialsCatalog?: number;
    materialEvalPassRate?: number;
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
  material_eval?: {
    passed?: boolean;
    material_name?: string;
    allowable_mpa?: number | null;
    mass_kg?: number | null;
    error?: string;
  };
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
