import { Canvas, useFrame } from "@react-three/fiber";
import {
  ContactShadows,
  Environment,
  Float,
  Grid,
  Html,
  Line,
  MeshTransmissionMaterial,
  OrbitControls,
  PerspectiveCamera,
  Sparkles,
} from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import type { GeometrySpec, Verdict } from "./types";

interface Props {
  geometry: GeometrySpec;
  ghosts?: Array<{ iteration: number; scale: number; geometry: GeometrySpec }>;
  showGhosts: boolean;
  fieldLens: boolean;
  verdict: Verdict;
  running: boolean;
}

const verdictColor: Record<Verdict, string> = {
  pass: "#5cf2b5",
  warn: "#f1b95b",
  fail: "#f05f71",
  idle: "#55b5ff",
};

function GeometryMesh({
  spec,
  color,
  opacity = 1,
  scale = 1,
  wireframe = false,
  fieldLens = false,
}: {
  spec: GeometrySpec;
  color: string;
  opacity?: number;
  scale?: number;
  wireframe?: boolean;
  fieldLens?: boolean;
}) {
  const dims: [number, number, number] = [
    spec.width ?? (spec.radius ?? 1.5) * 2,
    spec.height ?? (spec.radius ?? 1.5) * 2,
    spec.depth ?? (spec.radius ?? 1.5) * 2,
  ];
  const material = fieldLens ? (
    <meshStandardMaterial
      color="#ec6f55"
      emissive="#1e5aff"
      emissiveIntensity={0.38}
      metalness={0.55}
      roughness={0.25}
      wireframe={wireframe}
      transparent={opacity < 1}
      opacity={opacity}
    />
  ) : opacity < 0.95 ? (
    <meshBasicMaterial color={color} transparent opacity={opacity} wireframe={wireframe} depthWrite={false} />
  ) : (
    <MeshTransmissionMaterial
      color={color}
      thickness={0.65}
      roughness={0.26}
      transmission={0.18}
      metalness={0.68}
      chromaticAberration={0.025}
      ior={1.3}
    />
  );

  return (
    <mesh castShadow receiveShadow scale={scale}>
      {spec.kind === "cylinder" ? (
        <cylinderGeometry args={[spec.radius ?? 1.5, spec.radius ?? 1.5, spec.height ?? 3, 64]} />
      ) : spec.kind === "sphere" ? (
        <sphereGeometry args={[spec.radius ?? 1.7, 64, 32]} />
      ) : (
        <boxGeometry args={dims} />
      )}
      {material}
    </mesh>
  );
}

function Aura({ color, running }: { color: string; running: boolean }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const pulse = 1.025 + Math.sin(clock.elapsedTime * (running ? 4 : 1.4)) * 0.012;
    ref.current.scale.setScalar(pulse);
    ref.current.rotation.y = Math.sin(clock.elapsedTime * 0.18) * 0.025;
  });
  return (
    <mesh ref={ref}>
      <boxGeometry args={[4.88, 1.32, 2.68]} />
      <meshBasicMaterial color={color} transparent opacity={running ? 0.12 : 0.055} wireframe />
    </mesh>
  );
}

function ResidualRibbon({ running, color }: { running: boolean; color: string }) {
  const points = useMemo(
    () =>
      Array.from({ length: 90 }, (_, index) => {
        const t = (index / 89) * Math.PI * 2;
        return new THREE.Vector3(Math.cos(t) * 3.3, Math.sin(t * 3) * 0.22, Math.sin(t) * 2.05);
      }),
    [],
  );
  const group = useRef<THREE.Group>(null);
  useFrame((_, delta) => {
    if (group.current) group.current.rotation.y += delta * (running ? 0.38 : 0.05);
  });
  return (
    <group ref={group}>
      <Line points={points} color={color} lineWidth={running ? 1.7 : 0.65} transparent opacity={running ? 0.85 : 0.22} />
    </group>
  );
}

function DimensionLines({ spec }: { spec: GeometrySpec }) {
  const w = spec.width ?? (spec.radius ?? 1.5) * 2;
  const d = spec.depth ?? (spec.radius ?? 1.5) * 2;
  return (
    <group>
      <Line points={[[-w / 2, -1.15, d / 2 + 0.25], [w / 2, -1.15, d / 2 + 0.25]]} color="#718096" lineWidth={0.6} />
      <Html position={[0, -1.15, d / 2 + 0.25]} center distanceFactor={10}>
        <span className="dimension-label">{w.toFixed(2)} mm</span>
      </Html>
    </group>
  );
}

function Scene({ geometry, ghosts = [], showGhosts, fieldLens, verdict, running }: Props) {
  const color = verdictColor[verdict];
  return (
    <>
      <PerspectiveCamera makeDefault position={[7.8, 5.2, 8.8]} fov={38} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={5} maxDistance={22} />
      <ambientLight intensity={0.34} />
      <directionalLight castShadow intensity={3.6} position={[6, 9, 5]} color="#ddecff" />
      <pointLight intensity={36} distance={13} position={[-4, 2, -4]} color="#127dff" />
      <pointLight intensity={22} distance={10} position={[4, -1, 3]} color="#4ff2b6" />
      <Environment preset="city" environmentIntensity={0.42} />

      <Float speed={running ? 2.2 : 0.65} floatIntensity={running ? 0.12 : 0.035} rotationIntensity={0.018}>
        {showGhosts &&
          ghosts.map((ghost, index) => (
            <GeometryMesh
              key={ghost.iteration}
              spec={ghost.geometry}
              color={index === 2 ? "#edb45c" : "#4c8dff"}
              opacity={0.075 + index * 0.025}
              scale={ghost.scale}
              wireframe
            />
          ))}
        <GeometryMesh spec={geometry} color="#a9c8dd" fieldLens={fieldLens} />
        <Aura color={color} running={running} />
        <DimensionLines spec={geometry} />
      </Float>

      <ResidualRibbon running={running} color={color} />
      <Sparkles count={running ? 180 : 70} scale={[9, 5, 9]} size={1.1} speed={running ? 0.8 : 0.16} color={color} opacity={0.36} />
      <Grid
        position={[0, -1.7, 0]}
        args={[30, 30]}
        cellColor="#17222e"
        sectionColor="#263b4e"
        cellSize={0.5}
        sectionSize={2.5}
        fadeDistance={17}
        infiniteGrid
      />
      <ContactShadows position={[0, -1.62, 0]} opacity={0.5} scale={12} blur={2.5} far={5} color="#00040a" />
    </>
  );
}

export function GeometryViewport(props: Props) {
  return (
    <div className="viewport-canvas">
      <Canvas shadows gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }} dpr={[1, 1.75]}>
        <Scene {...props} />
      </Canvas>
      <div className="viewport-reticle" />
      <div className="viewport-axis">
        <span className="axis-x">X</span><span className="axis-y">Y</span><span className="axis-z">Z</span>
      </div>
    </div>
  );
}
