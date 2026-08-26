import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
g = 1.4

def _cone_angle_for(beta, M1):
    """Cone half-angle produced by a shock at wave angle beta, or None."""
    Mn1 = M1 * math.sin(beta)
    if Mn1 <= 1.0:
        return None
    delta = math.atan(2 / math.tan(beta) * (M1**2 * math.sin(beta)**2 - 1)
                      / (M1**2 * (g + math.cos(2*beta)) + 2))
    if delta <= 0:
        return None
    Mn2 = math.sqrt((1 + (g-1)/2 * Mn1**2) / (g * Mn1**2 - (g-1)/2))
    M2 = Mn2 / math.sin(beta - delta)
    Vp = 1 / math.sqrt(2 / ((g-1) * M2**2) + 1)
    y0 = [Vp * math.cos(beta - delta), -Vp * math.sin(beta - delta)]

    def rhs(th, y):
        vr, vth = y
        num = vth**2 * vr - (g-1)/2 * (1 - vr**2 - vth**2) * (2*vr + vth/math.tan(th))
        den = (g-1)/2 * (1 - vr**2 - vth**2) - vth**2
        return [vth, num/den]

    def ev(th, y):
        return y[1]
    ev.terminal = True
    ev.direction = 1
    s = solve_ivp(rhs, [beta, 1e-5], y0, events=ev, rtol=1e-10, atol=1e-13)
    if not s.t_events[0].size:
        return None
    return s.t_events[0][0], s.y[0][-1], y0, delta, Mn1

def cone_surface(M1, theta_c_deg):
    tc = math.radians(theta_c_deg)
    lo = math.asin(1.0/M1) + 1e-4
    hi = math.pi/2 - 1e-4
    betas = np.linspace(lo, hi, 400)
    prev_b = prev_f = None
    bracket = None
    for b in betas:
        r = _cone_angle_for(b, M1)
        if r is None:
            continue
        f = r[0] - tc
        if prev_f is not None and prev_f * f < 0:
            bracket = (prev_b, b)
            break
        prev_b, prev_f = b, f
    if bracket is None:
        return None
    beta = brentq(lambda b: _cone_angle_for(b, M1)[0] - tc, *bracket, xtol=1e-13)
    _, Vc, y0, delta, Mn1 = _cone_angle_for(beta, M1)
    p2_p1 = 1 + 2*g/(g+1) * (Mn1**2 - 1)
    V2 = math.hypot(*y0)
    pc_p2 = ((1 - Vc**2) / (1 - V2**2)) ** (g/(g-1))
    pc_p1 = p2_p1 * pc_p2
    Cp = (pc_p1 - 1) / (g/2 * M1**2)
    return math.degrees(beta), pc_p1, Cp

if __name__ == "__main__":
    for tc in (10, 15, 20):
        r = cone_surface(2.0, tc)
        if r:
            b, pr, cp = r
            print(f"M=2, cone {tc:2d} deg: shock {b:5.2f} deg  p_c/p_inf {pr:.4f}  "
                  f"Cp {cp:.4f}  p_c {101325*pr:8.0f} Pa")
