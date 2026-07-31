import { create } from "zustand";
import type { BootstrapData, BridgeEvent, GeometrySpec, PipelineResult, Theater } from "./types";

export const DEFAULT_GEOMETRY: GeometrySpec = {
  kind: "box",
  width: 4.8,
  height: 1.25,
  depth: 2.6,
  sculpt_offset: 0.04,
};

export const CATALOG_MATERIALS = [
  "Al 6061-T6",
  "Al 7075-T6",
  "Al-Li 2195",
  "Ti-6Al-4V",
  "Inconel 625",
  "Cu-Cr-Zr",
  "CFRP/Epoxy",
  "LI-900 Silica Tile",
  "PICA",
  "PEEK",
] as const;

export function geometryDefaultsForKind(kind: GeometrySpec["kind"]): GeometrySpec {
  switch (kind) {
    case "cylinder":
      return { kind, radius: 1.6, height: 3.5, sculpt_offset: 0.04 };
    case "sphere":
      return { kind, radius: 1.7, sculpt_offset: 0.02 };
    case "extrude":
      return {
        kind,
        height: 1.8,
        profile: [
          [0, 0],
          [2.2, 0],
          [2.0, 1.2],
          [0.7, 2.4],
          [1.8, 4.6],
          [0, 4.6],
        ],
        sculpt_offset: -0.01,
      };
    case "box":
    default:
      return { ...DEFAULT_GEOMETRY };
  }
}

interface StudioState {
  theater: Theater;
  setTheater: (theater: Theater) => void;
  bootstrap: BootstrapData | null;
  setBootstrap: (data: BootstrapData) => void;
  geometry: GeometrySpec;
  setGeometry: (geometry: Partial<GeometrySpec>) => void;
  replaceGeometry: (geometry: GeometrySpec) => void;
  solver: "fea" | "openfoam" | "mbd";
  setSolver: (solver: "fea" | "openfoam" | "mbd") => void;
  material: string;
  setMaterial: (material: string) => void;
  intentSummary: string | null;
  setIntentSummary: (summary: string | null) => void;
  running: boolean;
  progress: number;
  stage: string;
  eventLog: BridgeEvent[];
  result: PipelineResult | null;
  beginRun: () => void;
  applyEvent: (event: BridgeEvent) => void;
  completeRun: (result: PipelineResult) => void;
  failRun: (message: string) => void;
  resetStudio: () => void;
  showGhosts: boolean;
  toggleGhosts: () => void;
  fieldLens: boolean;
  toggleFieldLens: () => void;
  momentum: number;
}

export const useStudio = create<StudioState>((set) => ({
  theater: "forge",
  setTheater: (theater) => set({ theater }),
  bootstrap: null,
  setBootstrap: (bootstrap) => set({ bootstrap, momentum: bootstrap.stats.momentum }),
  geometry: { ...DEFAULT_GEOMETRY },
  setGeometry: (geometry) => set((state) => ({ geometry: { ...state.geometry, ...geometry } })),
  replaceGeometry: (geometry) => set({ geometry }),
  solver: "fea",
  setSolver: (solver) => set({ solver }),
  material: "Al 6061-T6",
  setMaterial: (material) => set({ material }),
  intentSummary: null,
  setIntentSummary: (intentSummary) => set({ intentSummary }),
  running: false,
  progress: 0,
  stage: "Ready",
  eventLog: [],
  result: null,
  beginRun: () => set({ running: true, progress: 0.02, stage: "Initializing", eventLog: [] }),
  applyEvent: (event) =>
    set((state) => ({
      eventLog: [...state.eventLog.slice(-39), event],
      progress: event.payload.progress ?? state.progress,
      stage: event.payload.message ?? event.payload.stage ?? state.stage,
    })),
  completeRun: (result) =>
    set((state) => ({
      result,
      running: false,
      progress: 1,
      stage: result.ok ? "Verified" : "Needs review",
      momentum: Math.min(100, state.momentum + result.momentumEarned),
    })),
  failRun: (message) =>
    set((state) => ({
      running: false,
      progress: 0,
      stage: message,
      eventLog: [...state.eventLog, { event: "run.error", payload: { level: "error", message } }],
    })),
  resetStudio: () =>
    set({
      geometry: { ...DEFAULT_GEOMETRY },
      solver: "fea",
      material: "Al 6061-T6",
      intentSummary: null,
      running: false,
      progress: 0,
      stage: "Ready",
      eventLog: [],
      result: null,
      showGhosts: true,
      fieldLens: false,
    }),
  showGhosts: true,
  toggleGhosts: () => set((state) => ({ showGhosts: !state.showGhosts })),
  fieldLens: false,
  toggleFieldLens: () => set((state) => ({ fieldLens: !state.fieldLens })),
  momentum: 0,
}));

export async function bridgeRequest<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
  if (window.lattice) return window.lattice.request<T>(method, params);
  return browserDemo(method, params) as T;
}

async function browserDemo(method: string, params: Record<string, unknown>): Promise<unknown> {
  await new Promise((resolve) => setTimeout(resolve, 300));
  if (method === "bootstrap") {
    return {
      appRoot: "~/.local/share/latticezero",
      repoRoot: "browser-demo",
      doctor: {
        native_ready: false,
        ready_backends: [],
        missing_backends: ["openfoam", "fea", "mbd"],
        probes: Object.fromEntries(
          ["openfoam", "fea", "mbd"].map((name) => [
            name,
            { backend: name, available: false, reason: "Desktop bridge required", details: {} },
          ]),
        ),
        runtime: {},
      },
      materials: {
        catalog_path: "browser-demo/materials_catalog.json",
        materials: 39,
        eval_report: "",
        suite_passed: 8,
        suite_failed: 2,
        pass_rate: 0.8,
      },
      stats: {
        runs: 18,
        verified: 13,
        promoted: 6,
        momentum: 72,
        modelVersion: "JEPA α.4",
        materialsCatalog: 39,
        materialEvalPassRate: 0.8,
      },
      recentRuns: [],
    };
  }
  if (method === "list_materials") {
    return {
      count: CATALOG_MATERIALS.length,
      materials: CATALOG_MATERIALS.map((name, index) => ({
        material_id: name.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
        name,
        category: ["aluminum", "aluminum", "aluminum", "titanium", "superalloy", "copper", "composite", "tps", "tps", "polymer"][index],
        density_kg_m3: 2700,
        youngs_modulus_gpa: 70,
        yield_mpa: 276,
        ultimate_mpa: 310,
        max_service_temp_k: 420,
        allowable_stress_mpa: 184,
      })),
    };
  }
  if (method === "material_eval_suite") {
    return {
      framework: "latticezero-material-eval/v1",
      cases: 10,
      passed: 8,
      failed: 2,
      pass_rate: 0.8,
      materials_catalog_size: 39,
      reports: [
        { case_id: "al6061_bracket", material_id: "al-6061-t6", material_name: "Al 6061-T6", passed: true, checks: [] },
        { case_id: "al_overstress", material_id: "al-6061-t6", material_name: "Al 6061-T6", passed: false, checks: [] },
      ],
    };
  }
  if (method === "latent_atlas") {
    return {
      modelVersion: "JEPA α.4",
      points: Array.from({ length: 72 }, (_, index) => ({
        id: `latent-${index}`,
        family: ["bracket", "fairing", "manifold", "linkage", "shell", "impeller"][index % 6],
        x: Math.cos(index * 0.61) * (2.4 + (index % 7) * 0.12),
        y: Math.sin(index * 0.61) * (2.4 + (index % 7) * 0.12),
        z: Math.sin(index * 0.24) * 1.8,
        score: 0.62 + (index % 31) / 100,
        verified: index % 4 !== 0,
      })),
    };
  }
  if (method === "run_pipeline") {
    const geometry = params.geometry as GeometrySpec;
    const residual = 0.00012;
    return {
      runId: `demo-${Date.now()}`,
      ok: true,
      run: { status: "verified", manifest: {}, provenance: { source: "latticezero" } },
      solver_result: {
        status: "optimal",
        objective: 141.8,
        residual,
        iterations: 37,
        metadata: { mode: "fallback", max_von_mises_mpa: 141.8, max_displacement_mm: 0.36 },
        artifacts: [],
      },
      verification: {
        name: "solid_verification",
        passed: true,
        findings: [],
        metrics: { volume: 15.62, face_count: 10, watertight: true, valid: true },
        backend: "cadquery",
      },
      artifacts: [],
      report_text: "solid_verification: PASSED",
      geometry,
      ghosts: [0.82, 0.91, 1.06].map((scale, iteration) => ({ iteration, scale, geometry })),
      metrics: {
        stress: 141.8,
        displacement: 0.36,
        volume: 15.62,
        faces: 10,
        watertight: true,
        mode: "fallback",
        objective: 141.8,
        residual,
        iterations: 37,
        material_eval_passed: true,
      },
      material_eval: { passed: true },
      momentumEarned: 12,
    };
  }
  if (method === "history") return { entries: [], total: 0, verified: 0 };
  if (method === "doctor") return (await browserDemo("bootstrap", {}) as BootstrapData).doctor;
  return { ok: true, decision: "demo_complete" };
}
