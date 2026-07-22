import type { GeometrySpec } from "./types";

export interface IntentPlan {
  geometry: GeometrySpec;
  solver: "fea" | "openfoam" | "mbd";
  material: string;
  summary: string;
}

function parsePressureBar(intent: string): number | null {
  const match = intent.match(/(\d+(?:\.\d+)?)\s*bar/i);
  return match ? Number(match[1]) : null;
}

function isNozzleIntent(text: string): boolean {
  return /(nozzle|rocket|combustor|chamber|thrust|injector|bell)/.test(text);
}

function isBracketIntent(text: string): boolean {
  return /(bracket|mount|support|fixture|linkage|arm|lug)/.test(text);
}

function isHousingIntent(text: string): boolean {
  return /(housing|fairing|shroud|duct|shell|cover|manifold|enclosure)/.test(text);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function nozzlePlan(intent: string): IntentPlan {
  const pressureBar = parsePressureBar(intent) ?? 250;
  const throatRadius = clamp(1.7 - Math.min(pressureBar, 400) / 800, 0.32, 1.5);
  const chamberRadius = throatRadius * 2.65;
  const exitRadius = throatRadius * 4.2;
  const chamberLength = Math.max(2.2, pressureBar / 130);
  const throatLength = 1.4;
  const bellLength = 4.6;
  const profile: [number, number][] = [
    [0, 0],
    [chamberRadius, 0],
    [chamberRadius * 1.02, chamberLength * 0.55],
    [throatRadius * 1.24, chamberLength + throatLength * 0.15],
    [throatRadius * 0.96, chamberLength + throatLength],
    [exitRadius * 0.92, chamberLength + throatLength + bellLength * 0.58],
    [exitRadius, chamberLength + throatLength + bellLength],
    [0, chamberLength + throatLength + bellLength],
  ];

  return {
    geometry: {
      kind: "extrude",
      profile,
      height: Math.max(1.2, throatRadius * 1.05),
      sculpt_offset: -0.018,
      features: [
        { op: "fillet", radius: Number((throatRadius * 0.09).toFixed(3)) },
        { op: "sculpt", distance: Number((throatRadius * 0.012).toFixed(3)) },
      ],
    },
    solver: "openfoam",
    material: "Ti-6Al-4V",
    summary: `Rocket nozzle profile generated from ~${pressureBar.toFixed(0)} bar chamber pressure`,
  };
}

function bracketPlan(intent: string): IntentPlan {
  const width = /wide|large|reinforced/i.test(intent) ? 8.8 : 6.2;
  const height = /thin|low-profile/i.test(intent) ? 1.35 : 1.9;
  const depth = /deep|stiff/i.test(intent) ? 4.2 : 3.1;
  const holeRadius = clamp(Math.min(width, depth) / 5.4, 0.45, 1.05);
  return {
    geometry: {
      kind: "box",
      width,
      height,
      depth,
      sculpt_offset: 0.05,
      features: [
        { op: "cut", tool: { kind: "cylinder", radius: holeRadius, height: depth * 1.25 } },
        { op: "fillet", radius: Number((Math.min(width, depth) * 0.04).toFixed(3)) },
        { op: "sculpt", distance: 0.08 },
      ],
    },
    solver: "fea",
    material: "Al 6061-T6",
    summary: "Bracket intent expanded into a cutout-and-fillet structural solid",
  };
}

function housingPlan(intent: string): IntentPlan {
  const radius = /large|wide|bulky/i.test(intent) ? 3.6 : 2.7;
  const height = /tall|long/i.test(intent) ? 7.8 : 5.2;
  return {
    geometry: {
      kind: "cylinder",
      radius,
      height,
      sculpt_offset: 0.035,
      features: [
        { op: "sculpt", distance: 0.055 },
        { op: "fillet", radius: Number((radius * 0.075).toFixed(3)) },
      ],
    },
    solver: "fea",
    material: "PA12-GF",
    summary: "Housing intent converted into a sculptable shell-like solid",
  };
}

export function planGeometryFromIntent(intent: string, baseGeometry?: GeometrySpec): IntentPlan | null {
  const normalized = intent.toLowerCase();
  if (!normalized.trim()) return null;

  if (isNozzleIntent(normalized)) return nozzlePlan(intent);
  if (isBracketIntent(normalized)) return bracketPlan(intent);
  if (isHousingIntent(normalized)) return housingPlan(intent);

  const base: GeometrySpec = baseGeometry ?? {
    kind: "box",
    width: 4.8,
    height: 1.25,
    depth: 2.6,
    sculpt_offset: 0.04,
  };
  const sculptDistance = /sculpt|organic|carve|form|blend/.test(normalized) ? 0.08 : 0.04;
  const filletRadius = clamp((base.width ?? 4.8) * 0.04, 0.12, 0.28);

  return {
    geometry: {
      ...base,
      sculpt_offset: base.sculpt_offset ?? sculptDistance / 2,
      features: [
        ...(base.features ?? []),
        { op: "sculpt", distance: sculptDistance },
        { op: "fillet", radius: Number(filletRadius.toFixed(3)) },
      ],
    },
    solver: "fea",
    material: "Al 6061-T6",
    summary: "General intent applied as sculpt and fillet features on the seed solid",
  };
}
