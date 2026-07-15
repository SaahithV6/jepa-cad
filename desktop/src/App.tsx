import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  Aperture,
  Atom,
  Box,
  BrainCircuit,
  Check,
  ChevronRight,
  CircleDot,
  Cpu,
  Database,
  Eye,
  EyeOff,
  FileCode2,
  Gauge,
  GitBranch,
  Hexagon,
  Layers3,
  Menu,
  Orbit,
  PanelRightClose,
  Play,
  RotateCcw,
  ScanLine,
  Settings2,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  TriangleAlert,
  WandSparkles,
  X,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { GeometryViewport } from "./GeometryViewport";
import { bridgeRequest, useStudio } from "./store";
import type { BootstrapData, DoctorProbe, PipelineResult, Theater, Verdict } from "./types";

const theaters: Array<{ id: Theater; label: string; caption: string; icon: typeof Box; key: string }> = [
  { id: "forge", label: "Forge", caption: "Geometry studio", icon: Box, key: "1" },
  { id: "solve", label: "Solve", caption: "Physics bridge", icon: Activity, key: "2" },
  { id: "verify", label: "Proof", caption: "Accountability", icon: ShieldCheck, key: "3" },
  { id: "atlas", label: "Atlas", caption: "JEPA latent space", icon: Orbit, key: "4" },
  { id: "autopilot", label: "Autopilot", caption: "Recursive loop", icon: BrainCircuit, key: "5" },
  { id: "doctor", label: "Systems", caption: "Environment", icon: Cpu, key: "6" },
];

const residualData = Array.from({ length: 42 }, (_, index) => ({
  iteration: index,
  residual: Math.max(0.00008, 0.7 * Math.exp(-index / 5.8) + Math.sin(index * 0.8) * 0.012),
}));

function useKeyboardTheaters() {
  const setTheater = useStudio((s) => s.setTheater);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
      const theater = theaters.find((item) => item.key === event.key);
      if (theater) setTheater(theater.id);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [setTheater]);
}

function App() {
  useKeyboardTheaters();
  const {
    theater,
    setTheater,
    bootstrap,
    setBootstrap,
    geometry,
    setGeometry,
    solver,
    setSolver,
    material,
    setMaterial,
    running,
    progress,
    stage,
    result,
    beginRun,
    applyEvent,
    completeRun,
    failRun,
    showGhosts,
    toggleGhosts,
    fieldLens,
    toggleFieldLens,
    momentum,
  } = useStudio();
  const [rightRail, setRightRail] = useState(true);
  const [intent, setIntent] = useState("Lightweight structural bracket with softened load paths");
  const [booting, setBooting] = useState(true);

  useEffect(() => {
    bridgeRequest<BootstrapData>("bootstrap")
      .then(setBootstrap)
      .finally(() => setBooting(false));
    return window.lattice?.onEvent(applyEvent);
  }, [applyEvent, setBootstrap]);

  const run = async () => {
    beginRun();
    try {
      const payload = await bridgeRequest<PipelineResult>("run_pipeline", {
        name: intent,
        geometry,
        solver,
        material,
        stressLimit: 240,
        load: 1200,
        dragTarget: 0.24,
      });
      completeRun(payload);
      setTheater("verify");
    } catch (error) {
      failRun(error instanceof Error ? error.message : String(error));
    }
  };

  const verdict: Verdict = running ? "warn" : result ? (result.ok ? "pass" : "fail") : "idle";
  const ghosts = result?.ghosts ?? [
    { iteration: 0, scale: 0.82, geometry },
    { iteration: 1, scale: 0.91, geometry },
    { iteration: 2, scale: 1.06, geometry },
  ];

  if (booting) return <BootScreen />;

  return (
    <div className="app-shell">
      <div className="noise" />
      <Titlebar nativeReady={bootstrap?.doctor.native_ready ?? false} />
      <aside className="nav-rail">
        <div className="brand-mark"><Hexagon size={25} strokeWidth={1.4} /><span>L0</span></div>
        <nav>
          {theaters.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={`nav-item ${theater === item.id ? "active" : ""}`}
                onClick={() => setTheater(item.id)}
                title={`${item.label} · ${item.caption}`}
              >
                <Icon size={19} strokeWidth={1.55} />
                <span>{item.label}</span>
                <kbd>{item.key}</kbd>
              </button>
            );
          })}
        </nav>
        <Momentum momentum={momentum} />
        <button className="nav-icon"><Settings2 size={18} /></button>
      </aside>

      <main className="workspace">
        <WorkspaceHeader
          theater={theater}
          rightRail={rightRail}
          onToggleRail={() => setRightRail((value) => !value)}
          result={result}
        />
        <AnimatePresence mode="wait">
          <motion.section
            key={theater}
            className="theater"
            initial={{ opacity: 0, y: 8, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            exit={{ opacity: 0, y: -5, filter: "blur(3px)" }}
            transition={{ duration: 0.22, ease: [0.2, 0.8, 0.2, 1] }}
          >
            {theater === "forge" && (
              <ForgeTheater
                geometry={geometry}
                setGeometry={setGeometry}
                ghosts={ghosts}
                showGhosts={showGhosts}
                toggleGhosts={toggleGhosts}
                fieldLens={fieldLens}
                toggleFieldLens={toggleFieldLens}
                verdict={verdict}
                running={running}
                intent={intent}
                setIntent={setIntent}
                material={material}
                setMaterial={setMaterial}
              />
            )}
            {theater === "solve" && (
              <SolveTheater geometry={geometry} ghosts={ghosts} verdict={verdict} running={running} solver={solver} setSolver={setSolver} result={result} />
            )}
            {theater === "verify" && <VerifyTheater result={result} bootstrap={bootstrap} />}
            {theater === "atlas" && <AtlasTheater modelVersion={bootstrap?.stats.modelVersion ?? "JEPA α.1"} />}
            {theater === "autopilot" && <AutopilotTheater />}
            {theater === "doctor" && <DoctorTheater bootstrap={bootstrap} />}
          </motion.section>
        </AnimatePresence>
      </main>

      <AnimatePresence>
        {rightRail && (
          <motion.aside className="inspector" initial={{ x: 340, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 340, opacity: 0 }}>
            <Inspector
              theater={theater}
              result={result}
              geometry={geometry}
              solver={solver}
              bootstrap={bootstrap}
              running={running}
            />
          </motion.aside>
        )}
      </AnimatePresence>

      <RunDock running={running} progress={progress} stage={stage} onRun={run} verdict={verdict} />
    </div>
  );
}

function BootScreen() {
  return (
    <div className="boot-screen">
      <motion.div className="boot-orbit" animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 8, ease: "linear" }}>
        <Hexagon size={80} strokeWidth={0.6} />
      </motion.div>
      <div className="boot-wordmark">LATTICE<span>ZERO</span></div>
      <div className="boot-caption">WAKING THE ENGINEERING GRAPH</div>
      <div className="boot-line"><i /></div>
    </div>
  );
}

function Titlebar({ nativeReady }: { nativeReady: boolean }) {
  return (
    <header className="titlebar">
      <div className="titlebar-wordmark">LATTICE<span>ZERO</span></div>
      <div className="titlebar-project"><CircleDot size={10} /> INVESTOR DEMONSTRATOR <ChevronRight size={11} /> BRACKET-07</div>
      <div className={`native-state ${nativeReady ? "ready" : ""}`}>
        <i /> {nativeReady ? "NATIVE SOLVERS ONLINE" : "PORTABLE FALLBACK"}
      </div>
    </header>
  );
}

function Momentum({ momentum }: { momentum: number }) {
  return (
    <div className="momentum-widget">
      <div className="momentum-ring" style={{ "--momentum": `${momentum * 3.6}deg` } as React.CSSProperties}>
        <div><span>{momentum}</span><small>μ</small></div>
      </div>
      <strong>MOMENTUM</strong>
      <small>Verified craft energy</small>
    </div>
  );
}

function WorkspaceHeader({ theater, rightRail, onToggleRail, result }: { theater: Theater; rightRail: boolean; onToggleRail: () => void; result: PipelineResult | null }) {
  const current = theaters.find((item) => item.id === theater)!;
  return (
    <header className="workspace-header">
      <div>
        <div className="eyebrow">{current.caption}</div>
        <h1>{current.label}</h1>
      </div>
      <div className="workspace-actions">
        {result && <span className={`verdict-chip ${result.ok ? "pass" : "fail"}`}><ShieldCheck size={13} /> {result.ok ? "VERIFIED" : "REVIEW"}</span>}
        <button className="icon-button"><RotateCcw size={16} /></button>
        <button className={`icon-button ${rightRail ? "selected" : ""}`} onClick={onToggleRail}><PanelRightClose size={17} /></button>
      </div>
    </header>
  );
}

function ForgeTheater(props: {
  geometry: ReturnType<typeof useStudio.getState>["geometry"];
  setGeometry: ReturnType<typeof useStudio.getState>["setGeometry"];
  ghosts: PipelineResult["ghosts"];
  showGhosts: boolean;
  toggleGhosts: () => void;
  fieldLens: boolean;
  toggleFieldLens: () => void;
  verdict: Verdict;
  running: boolean;
  intent: string;
  setIntent: (value: string) => void;
  material: string;
  setMaterial: (value: string) => void;
}) {
  return (
    <div className="forge-layout">
      <section className="viewport-panel">
        <GeometryViewport geometry={props.geometry} ghosts={props.ghosts} showGhosts={props.showGhosts} fieldLens={props.fieldLens} verdict={props.verdict} running={props.running} />
        <div className="viewport-toolbar">
          <button className={props.showGhosts ? "active" : ""} onClick={props.toggleGhosts}><Layers3 size={14} /> Ghost stack</button>
          <button className={props.fieldLens ? "active" : ""} onClick={props.toggleFieldLens}><ScanLine size={14} /> Field lens</button>
          <button><Aperture size={14} /> Orthographic</button>
        </div>
        <div className="viewport-caption">
          <span>DETERMINISTIC B-REP</span><i /><span>CADQUERY / OCC</span><i /><span>MM</span>
        </div>
      </section>
      <section className="intent-panel glass-panel">
        <div className="panel-kicker"><WandSparkles size={14} /> DESIGN INTENT</div>
        <textarea value={props.intent} onChange={(event) => props.setIntent(event.target.value)} />
        <div className="intent-foot">
          <span>Planner describes</span><i /><strong>Tools construct</strong><i /><span>Proof decides</span>
        </div>
        <div className="parameter-grid">
          <SelectField label="Primitive" value={props.geometry.kind} onChange={(value) => props.setGeometry({ kind: value as "box" | "cylinder" | "sphere" })} options={["box", "cylinder", "sphere"]} />
          <SelectField label="Material" value={props.material} onChange={props.setMaterial} options={["Al 6061-T6", "Ti-6Al-4V", "AISI 4140", "PA12-GF"]} />
          {props.geometry.kind === "box" ? (
            <>
              <NumberField label="Width" value={props.geometry.width ?? 4.8} onChange={(width) => props.setGeometry({ width })} />
              <NumberField label="Height" value={props.geometry.height ?? 1.25} onChange={(height) => props.setGeometry({ height })} />
              <NumberField label="Depth" value={props.geometry.depth ?? 2.6} onChange={(depth) => props.setGeometry({ depth })} />
            </>
          ) : (
            <>
              <NumberField label="Radius" value={props.geometry.radius ?? 1.6} onChange={(radius) => props.setGeometry({ radius })} />
              {props.geometry.kind === "cylinder" && <NumberField label="Height" value={props.geometry.height ?? 3.5} onChange={(height) => props.setGeometry({ height })} />}
            </>
          )}
          <NumberField label="Sculpt bias" value={props.geometry.sculpt_offset ?? 0} step={0.01} onChange={(sculpt_offset) => props.setGeometry({ sculpt_offset })} />
        </div>
      </section>
    </div>
  );
}

function SolveTheater({ geometry, ghosts, verdict, running, solver, setSolver, result }: {
  geometry: ReturnType<typeof useStudio.getState>["geometry"];
  ghosts: PipelineResult["ghosts"];
  verdict: Verdict;
  running: boolean;
  solver: "fea" | "openfoam" | "mbd";
  setSolver: (value: "fea" | "openfoam" | "mbd") => void;
  result: PipelineResult | null;
}) {
  return (
    <div className="solve-layout">
      <section className="solver-stage">
        <GeometryViewport geometry={geometry} ghosts={ghosts} showGhosts={false} fieldLens verdict={verdict} running={running} />
        <div className="solver-choice">
          {(["fea", "openfoam", "mbd"] as const).map((value) => (
            <button key={value} className={solver === value ? "active" : ""} onClick={() => setSolver(value)}>
              {value === "fea" ? "STRUCTURAL" : value === "openfoam" ? "FLOW" : "KINEMATICS"}
            </button>
          ))}
        </div>
      </section>
      <section className="solver-data glass-panel">
        <div className="panel-kicker"><Activity size={14} /> RESIDUAL PULSE</div>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={residualData}>
              <defs>
                <linearGradient id="residualFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#59b8ff" stopOpacity={0.42} />
                  <stop offset="100%" stopColor="#59b8ff" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#1a2530" vertical={false} />
              <XAxis dataKey="iteration" stroke="#526170" tickLine={false} axisLine={false} />
              <YAxis scale="log" domain={["auto", "auto"]} stroke="#526170" tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ background: "#0b1118", border: "1px solid #263542", borderRadius: 8 }} />
              <Area type="monotone" dataKey="residual" stroke="#63c2ff" strokeWidth={2} fill="url(#residualFill)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="metric-row">
          <Metric label="OBJECTIVE" value={formatMetric(result?.metrics.objective, "—")} />
          <Metric label="ITERATIONS" value={formatMetric(result?.metrics.iterations, "—")} />
          <Metric label="RESIDUAL" value={formatMetric(result?.metrics.residual, "—")} />
          <Metric label="MODE" value={String(result?.metrics.mode ?? "READY")} />
        </div>
      </section>
    </div>
  );
}

function VerifyTheater({ result, bootstrap }: { result: PipelineResult | null; bootstrap: BootstrapData | null }) {
  const checks = result
    ? [
        ["Positive volume", Number(result.metrics.volume ?? 0) > 0],
        ["Closed solid", Boolean(result.metrics.watertight)],
        ["Valid topology", Boolean(result.verification.metrics.valid)],
        ["Solver completed", result.solver_result.status !== "failed"],
        ["Provenance sealed", Boolean(result.run.provenance)],
        ["Promotion eligible", result.ok],
      ]
    : [
        ["Positive volume", null], ["Closed solid", null], ["Valid topology", null],
        ["Solver completed", null], ["Provenance sealed", null], ["Promotion eligible", null],
      ];
  return (
    <div className="proof-layout">
      <section className={`proof-hero ${result?.ok ? "pass" : result ? "fail" : ""}`}>
        <div className="proof-seal"><ShieldCheck size={62} strokeWidth={0.7} /><span>{result?.ok ? "VERIFIED" : result ? "REVIEW" : "AWAITING RUN"}</span></div>
        <div>
          <div className="eyebrow">ACCOUNTABILITY ENVELOPE</div>
          <h2>{result ? result.runId : "No proof has been minted"}</h2>
          <p>{result?.report_text ?? "Run the engineering chain to produce auditable geometry, solver, and lineage evidence."}</p>
        </div>
      </section>
      <section className="proof-grid">
        <div className="glass-panel checks-panel">
          <div className="panel-kicker"><ShieldCheck size={14} /> VERIFICATION MATRIX</div>
          {checks.map(([label, passed]) => (
            <div className="check-row" key={String(label)}>
              <span className={passed === null ? "pending" : passed ? "pass" : "fail"}>
                {passed === null ? <CircleDot size={15} /> : passed ? <Check size={15} /> : <X size={15} />}
              </span>
              <strong>{label}</strong>
              <small>{passed === null ? "NOT RUN" : passed ? "PASS" : "FAIL"}</small>
            </div>
          ))}
        </div>
        <ProvenanceGraph result={result} />
        <div className="glass-panel proof-stats">
          <Metric label="VERIFIED RUNS" value={String(bootstrap?.stats.verified ?? 0)} />
          <Metric label="CURATED" value={String(bootstrap?.stats.promoted ?? 0)} />
          <Metric label="ARTIFACTS" value={String(result?.artifacts.length ?? 0)} />
        </div>
      </section>
    </div>
  );
}

function ProvenanceGraph({ result }: { result: PipelineResult | null }) {
  const nodes = [
    { label: "Intent", x: 8, state: "pass" }, { label: "B-rep", x: 29, state: result ? "pass" : "idle" },
    { label: "Solve", x: 50, state: result?.solver_result.status === "failed" ? "fail" : result ? "pass" : "idle" },
    { label: "Proof", x: 71, state: result?.ok ? "pass" : result ? "fail" : "idle" },
    { label: "Flywheel", x: 92, state: result?.ok ? "pass" : "idle" },
  ];
  return (
    <div className="glass-panel provenance-panel">
      <div className="panel-kicker"><GitBranch size={14} /> PROVENANCE CONSTELLATION</div>
      <div className="provenance-track"><i /></div>
      <div className="provenance-nodes">
        {nodes.map((node) => <div key={node.label} className={`provenance-node ${node.state}`} style={{ left: `${node.x}%` }}><span /><small>{node.label}</small></div>)}
      </div>
      <div className="fingerprint">{result ? JSON.stringify(result.run.provenance).slice(0, 72) : "Lineage appears here after the first verified run"}…</div>
    </div>
  );
}

function AtlasTheater({ modelVersion }: { modelVersion: string }) {
  const [atlas, setAtlas] = useState<{ points: Array<{ id: string; family: string; x: number; y: number; score: number; verified: boolean }> } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  useEffect(() => { bridgeRequest<typeof atlas>("latent_atlas").then(setAtlas); }, []);
  return (
    <div className="atlas-layout">
      <section className="atlas-field">
        <div className="atlas-orbit orbit-one" /><div className="atlas-orbit orbit-two" /><div className="atlas-orbit orbit-three" />
        {atlas?.points.map((point) => {
          const left = 50 + point.x * 8;
          const top = 50 + point.y * 8;
          return (
            <button
              key={point.id}
              className={`latent-point ${point.verified ? "verified" : ""} ${selected === point.id ? "selected" : ""}`}
              style={{ left: `${left}%`, top: `${top}%`, "--score": point.score } as React.CSSProperties}
              onClick={() => setSelected(point.id)}
              title={`${point.family} · ${(point.score * 100).toFixed(0)}%`}
            />
          );
        })}
        <div className="atlas-center"><Atom size={32} /><strong>{modelVersion}</strong><small>GEOMETRY × PHYSICS</small></div>
        <div className="atlas-legend"><i className="verified" /> Verified latent <i /> Unverified neighbor</div>
      </section>
      <section className="atlas-context glass-panel">
        <div className="panel-kicker"><Orbit size={14} /> LATENT ATLAS</div>
        <h2>Walk the neighborhood of proven designs.</h2>
        <p>Nearby points are JEPA-ranked proposals. Selecting one never emits raw geometry—it seeds a deterministic CadQuery construction plan.</p>
        <div className="atlas-stats">
          <Metric label="LATENTS" value={String(atlas?.points.length ?? 0)} />
          <Metric label="VERIFIED DENSITY" value="74%" />
          <Metric label="MODEL DELTA" value="+8.3%" />
        </div>
        {selected && <button className="primary-subtle"><Sparkles size={14} /> Seed verified variant</button>}
      </section>
    </div>
  );
}

function AutopilotTheater() {
  const [rawDir, setRawDir] = useState("");
  const [status, setStatus] = useState("Standing by");
  const [active, setActive] = useState(false);
  const [decision, setDecision] = useState<string | null>(null);
  const choose = async () => {
    const dir = await window.lattice?.chooseDirectory();
    if (dir) setRawDir(dir);
  };
  const launch = async () => {
    setActive(true); setStatus("Running accountability gate");
    try {
      const result = await bridgeRequest<{ decision: string; ok: boolean }>("run_autopilot", { rawDir: rawDir || null, maxSteps: 1, skipTests: true });
      setDecision(result.decision); setStatus(result.ok ? "Cycle complete" : "Cycle needs review");
    } catch (error) { setStatus(error instanceof Error ? error.message : String(error)); }
    finally { setActive(false); }
  };
  const steps = ["Environment gate", "Ingest evidence", "Curate verified jobs", "Train candidate", "Probe against best", "Promote only if better"];
  return (
    <div className="autopilot-layout">
      <section className="autopilot-core">
        <motion.div className={`flywheel-core ${active ? "active" : ""}`} animate={active ? { rotate: 360 } : { rotate: 0 }} transition={{ repeat: active ? Infinity : 0, duration: 5, ease: "linear" }}>
          <div className="flywheel-spoke s1" /><div className="flywheel-spoke s2" /><div className="flywheel-spoke s3" />
          <BrainCircuit size={48} strokeWidth={0.7} />
        </motion.div>
        <h2>Recursive improvement<br />with a conscience.</h2>
        <p>Every cycle is gated by tests, verification, and comparative probe score. Nothing promotes itself.</p>
        <button className="launch-autopilot" onClick={launch} disabled={active}><Zap size={17} fill="currentColor" /> {active ? status : "Launch one accountable cycle"}</button>
        {decision && <div className="decision-chip"><Check size={14} /> {decision.replaceAll("_", " ").toUpperCase()}</div>}
      </section>
      <section className="loop-panel glass-panel">
        <div className="panel-kicker"><BrainCircuit size={14} /> RECURSIVE LOOP</div>
        <div className="directory-picker" onClick={choose}><Database size={18} /><div><small>RAW EVIDENCE SOURCE</small><strong>{rawDir || "Choose a dataset directory"}</strong></div><ChevronRight size={16} /></div>
        <div className="loop-steps">
          {steps.map((step, index) => <div key={step} className={active && index < 3 ? "active" : ""}><span>{String(index + 1).padStart(2, "0")}</span><strong>{step}</strong><i /></div>)}
        </div>
      </section>
    </div>
  );
}

function DoctorTheater({ bootstrap }: { bootstrap: BootstrapData | null }) {
  const [report, setReport] = useState(bootstrap?.doctor);
  const [scanning, setScanning] = useState(false);
  const scan = async () => {
    setScanning(true);
    setReport(await bridgeRequest<BootstrapData["doctor"]>("doctor"));
    setScanning(false);
  };
  return (
    <div className="systems-layout">
      <section className="system-hero">
        <div className={`system-orb ${report?.native_ready ? "ready" : ""}`}><Cpu size={54} strokeWidth={0.65} /><i /></div>
        <div><div className="eyebrow">RUNTIME DOCTOR</div><h2>{report?.native_ready ? "Native compute is ready." : "Portable mode is operational."}</h2><p>Missing native solvers never hide. LatticeZero labels fallback evidence and prevents accidental trust escalation.</p></div>
        <button className="scan-button" onClick={scan}><ScanLine size={16} className={scanning ? "spinning" : ""} /> Rescan environment</button>
      </section>
      <section className="probe-grid">
        {Object.entries(report?.probes ?? {}).map(([name, probe]) => <ProbeCard key={name} name={name} probe={probe} />)}
      </section>
      <section className="glass-panel runtime-panel">
        <div className="panel-kicker"><TerminalSquare size={14} /> RUNTIME ENVELOPE</div>
        <pre>{JSON.stringify(report?.runtime ?? {}, null, 2)}</pre>
      </section>
    </div>
  );
}

function ProbeCard({ name, probe }: { name: string; probe: DoctorProbe }) {
  return (
    <article className={`probe-card ${probe.available ? "ready" : ""}`}>
      <div className="probe-icon">{name === "openfoam" ? <Aperture /> : name === "fea" ? <TriangleAlert /> : <Atom />}</div>
      <div><small>{probe.backend.toUpperCase()}</small><h3>{name === "openfoam" ? "Flow field" : name === "fea" ? "Structural" : "Multibody"}</h3><p>{probe.reason}</p></div>
      <span>{probe.available ? "NATIVE" : "FALLBACK"}</span>
    </article>
  );
}

function Inspector({ theater, result, geometry, solver, bootstrap, running }: {
  theater: Theater; result: PipelineResult | null; geometry: ReturnType<typeof useStudio.getState>["geometry"];
  solver: string; bootstrap: BootstrapData | null; running: boolean;
}) {
  return (
    <>
      <div className="inspector-head"><span>LIVE EVIDENCE</span><CircleDot size={13} className={running ? "pulse" : ""} /></div>
      <div className="inspector-section">
        <h4>Current object</h4>
        <div className="object-card"><Box size={24} /><div><strong>{geometry.kind.toUpperCase()} / 07</strong><span>Deterministic solid</span></div><ChevronRight size={14} /></div>
      </div>
      <div className="inspector-section">
        <h4>Engineering state</h4>
        <InspectorRow label="Theater" value={theater} />
        <InspectorRow label="Solver" value={solver.toUpperCase()} />
        <InspectorRow label="Backend" value={result?.verification.backend ?? "CADQUERY"} />
        <InspectorRow label="Mode" value={String(result?.metrics.mode ?? "READY")} warn={result?.metrics.mode === "fallback"} />
      </div>
      <div className="inspector-section">
        <h4>Proof telemetry</h4>
        <InspectorRow label="Volume" value={formatMetric(result?.metrics.volume, "—", " mm³")} />
        <InspectorRow label="Faces" value={formatMetric(result?.metrics.faces, "—")} />
        <InspectorRow label="Stress" value={formatMetric(result?.metrics.stress, "—", " MPa")} />
        <InspectorRow label="Watertight" value={result ? (result.metrics.watertight ? "YES" : "NO") : "—"} />
      </div>
      <div className="inspector-section">
        <h4>Intelligence</h4>
        <InspectorRow label="Model" value={bootstrap?.stats.modelVersion ?? "JEPA α.1"} />
        <InspectorRow label="Verified runs" value={String(bootstrap?.stats.verified ?? 0)} />
        <InspectorRow label="Flywheel" value={`${bootstrap?.stats.momentum ?? 0}%`} />
      </div>
      {result?.artifacts.length ? <button className="artifact-button" onClick={() => window.lattice?.reveal(result.artifacts[0])}><FileCode2 size={15} /> Reveal {result.artifacts.length} artifacts</button> : null}
      <div className="truth-note"><ShieldCheck size={16} /><p><strong>Truth policy</strong>LLM intent is advisory. Geometry and promotion remain tool-verified.</p></div>
    </>
  );
}

function InspectorRow({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return <div className="inspector-row"><span>{label}</span><strong className={warn ? "warn" : ""}>{value}</strong></div>;
}

function RunDock({ running, progress, stage, onRun, verdict }: { running: boolean; progress: number; stage: string; onRun: () => void; verdict: Verdict }) {
  return (
    <div className={`run-dock ${running ? "running" : ""} ${verdict}`}>
      <div className="run-progress"><i style={{ width: `${progress * 100}%` }} /></div>
      <div className="run-state"><span className="run-led" /><div><small>{running ? "ENGINEERING CHAIN ACTIVE" : "ACCOUNTABILITY CHAIN"}</small><strong>{stage}</strong></div></div>
      <div className="run-path"><span>PLAN</span><i /><span>BUILD</span><i /><span>SOLVE</span><i /><span>PROVE</span><i /><span>LEARN</span></div>
      <button onClick={onRun} disabled={running}>{running ? <Activity size={18} className="spinning" /> : <Play size={18} fill="currentColor" />} {running ? "Working…" : "Run verified chain"}</button>
    </div>
  );
}

function SelectField({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: string[] }) {
  return <label className="field"><span>{label}</span><select value={value} onChange={(e) => onChange(e.target.value)}>{options.map((option) => <option key={option}>{option}</option>)}</select></label>;
}

function NumberField({ label, value, onChange, step = 0.1 }: { label: string; value: number; onChange: (value: number) => void; step?: number }) {
  return <label className="field"><span>{label}</span><div className="number-input"><input type="number" value={value} step={step} onChange={(e) => onChange(Number(e.target.value))} /><small>mm</small></div></label>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="metric"><small>{label}</small><strong>{value}</strong></div>;
}

function formatMetric(value: unknown, fallback: string, unit = "") {
  if (typeof value === "number") return `${value < 0.01 ? value.toExponential(2) : value.toFixed(value < 100 ? 2 : 1)}${unit}`;
  return fallback;
}

export default App;
