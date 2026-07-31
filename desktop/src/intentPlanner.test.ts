import { describe, expect, it } from "vitest";
import { planFromAtlasFamily, planGeometryFromIntent } from "./intentPlanner";

describe("planGeometryFromIntent", () => {
  it("builds a nozzle extrude with a closed profile", () => {
    const plan = planGeometryFromIntent("250 bar rocket nozzle thrust chamber");
    expect(plan).not.toBeNull();
    expect(plan!.geometry.kind).toBe("extrude");
    expect(plan!.geometry.profile!.length).toBeGreaterThan(4);
    expect(plan!.solver).toBe("openfoam");
    expect(plan!.material).toBe("Ti-6Al-4V");
  });

  it("builds a bracket with catalog material", () => {
    const plan = planGeometryFromIntent("lightweight structural bracket mount");
    expect(plan!.geometry.kind).toBe("box");
    expect(plan!.material).toBe("Al 6061-T6");
    expect(plan!.solver).toBe("fea");
  });

  it("maps housing to a known catalog polymer", () => {
    const plan = planGeometryFromIntent("avionics housing enclosure shell");
    expect(plan!.geometry.kind).toBe("cylinder");
    expect(plan!.material).toBe("PEEK");
  });
});

describe("planFromAtlasFamily", () => {
  it("seeds deterministic geometry for atlas families", () => {
    expect(planFromAtlasFamily("impeller").geometry.kind).toBe("extrude");
    expect(planFromAtlasFamily("bracket").geometry.kind).toBe("box");
    expect(planFromAtlasFamily("fairing").geometry.kind).toBe("cylinder");
  });
});
