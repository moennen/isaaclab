# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Material space sweep for VBD cube grasping physics.

Calls generate_sequences.py as a subprocess for each probe point with
--num_sequences 4 --reachable_ratio 1.0 --success_ratio 1.0 so all
generated sequences are reachable_success.  Parses peak_z from stdout
and reports which parameter set gives reliable lifting.

Root-cause analysis (2026-04-14)
---------------------------------
Normal contact force per particle = soft_ke × particle_radius = 1e4×0.015 = 150N.
Finger actuator effort limit = 100N → fingers spring open → cube drops.

Fix space:
  (A) Lower soft_ke → smaller normal force per contact
  (B) Raise finger_ke + finger_effort → actuator can resist the contact load
  (C) Raise friction (soft_mu × contact_mu) → better grip per unit penetration
  (D) Lower density → lighter cube (easier to lift)

Effective friction coefficient: mu_eff = sqrt(soft_mu × contact_mu)
Max normal force per particle:  F_n = soft_ke × particle_radius

Usage
-----
cd source/isaaclab_tasks/.../franka_vbd_cube_pick
micromamba run -n env_isaaclab python scripts/sweep_physics.py [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).parent
_GEN_SCRIPT = _SCRIPT_DIR / "generate_sequences.py"
_LIFT_HEIGHT = 0.20   # m — match sequence_schema DEFAULT_CONFIG

PARTICLE_RADIUS = 0.015  # m


@dataclass
class Probe:
    label:         str
    soft_ke:       float
    soft_mu:       float
    contact_mu:    float
    density:       float
    finger_ke:     float
    finger_effort: float

    @property
    def mu_eff(self) -> float:
        return (self.soft_ke * 0 + (self.soft_mu * self.contact_mu) ** 0.5)  # sqrt(soft_mu × contact_mu)

    @property
    def max_normal_per_particle(self) -> float:
        return self.soft_ke * PARTICLE_RADIUS

    def to_args(self) -> list[str]:
        return [
            f"--soft-ke={self.soft_ke}",
            f"--soft-mu={self.soft_mu}",
            f"--contact-mu={self.contact_mu}",
            f"--density={self.density}",
            f"--finger-ke={self.finger_ke}",
            f"--finger-effort={self.finger_effort}",
        ]


# ---------------------------------------------------------------------------
# Sweep design
# ---------------------------------------------------------------------------
# Force balance for a 0.05m cube:
#   weight = density × (0.05)³ × 9.81
#   max grip force ≈ min(finger_effort, soft_ke × radius) × mu_eff × n_contacts
#
# Finger springs open if contact normal load > finger_effort.
# Contact normal load per particle = soft_ke × penetration ≈ soft_ke × particle_radius
# → need: finger_effort > soft_ke × particle_radius
#          100 > 1e4 × 0.015 = 150  ← CURRENT BUG: finger effort too small
#
# All probes aim for: (soft_ke × particle_radius) << finger_effort
#   or equivalently soft_ke << finger_effort / particle_radius

SWEEPS = [
    # A: CURRENT BASELINE (finger force 100N < contact 150N/particle → fails)
    Probe("A:baseline",     soft_ke=1e4, soft_mu=1.5, contact_mu=0.75, density=400, finger_ke=100,  finger_effort=100),

    # B: reduce soft_ke so finger can resist contact load (1e3 × 0.015 = 15N < 100N ✓)
    Probe("B:lo_ke",        soft_ke=1e3, soft_mu=2.0, contact_mu=1.5,  density=200, finger_ke=100,  finger_effort=100),

    # C: increase finger effort to match current ke (finger 500N > 150N/particle ✓)
    Probe("C:hi_feffort",   soft_ke=1e4, soft_mu=2.0, contact_mu=1.5,  density=200, finger_ke=500,  finger_effort=500),

    # D: balanced — medium ke + strong fingers + high friction + light cube
    Probe("D:balanced",     soft_ke=3e3, soft_mu=3.0, contact_mu=2.0,  density=150, finger_ke=300,  finger_effort=200),

    # E: very low ke (minimal disturbance to finger) + extreme friction
    Probe("E:lo_ke_hi_fr",  soft_ke=5e2, soft_mu=5.0, contact_mu=4.0,  density=100, finger_ke=200,  finger_effort=200),

    # F: hi ke + very strong fingers + ultra-high mu
    Probe("F:hi_ke+fstr",   soft_ke=5e4, soft_mu=3.0, contact_mu=3.0,  density=100, finger_ke=2000, finger_effort=1000),
]


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_probe(probe: Probe, dry_run: bool = False) -> dict:
    """Run generate_sequences.py for this probe and parse peak_z from stdout."""
    import math
    mu_eff = math.sqrt(probe.soft_mu * probe.contact_mu)
    max_n  = probe.soft_ke * PARTICLE_RADIUS
    mass_g = probe.density * (0.05 ** 3) * 1000
    print(f"\n{'='*65}")
    print(f"Probe {probe.label}")
    print(f"  soft_ke={probe.soft_ke:.0e}  soft_mu={probe.soft_mu}  contact_mu={probe.contact_mu}")
    print(f"  density={probe.density} kg/m³  mass≈{mass_g:.0f}g")
    print(f"  finger_ke={probe.finger_ke}  finger_effort={probe.finger_effort}N")
    print(f"  mu_eff=√({probe.soft_mu}×{probe.contact_mu})={mu_eff:.2f}"
          f"  max_F_normal/particle={max_n:.0f}N"
          f"  {'OK ✓' if probe.finger_effort > max_n else 'FINGER OPENS ✗'}")
    print(f"{'='*65}")

    cmd = [
        sys.executable, str(_GEN_SCRIPT),
        "--num_sequences", "4",
        "--num-worlds", "4",
        "--reachable_ratio", "1.0",   # all reachable
        "--success_ratio",  "1.0",    # all success
        "--output", "/dev/null",      # discard JSON — we only need peak_z
        "--seed", "123",
    ] + probe.to_args()

    if dry_run:
        print(f"  DRY RUN: {' '.join(cmd)}")
        return {"label": probe.label, "probe": probe, "peak_z": [], "hit": 0, "elapsed": 0}

    t0 = time.time()
    stdout_lines = []
    stderr_lines = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=str(_SCRIPT_DIR),
    )

    import threading

    def drain(pipe, store, prefix=""):
        for line in pipe:
            store.append(line)
            print(f"  {prefix}{line}", end="", flush=True)
        pipe.close()

    t_out = threading.Thread(target=drain, args=(proc.stdout, stdout_lines))
    t_err = threading.Thread(target=drain, args=(proc.stderr, stderr_lines, "[err] "))
    t_out.start(); t_err.start()
    proc.wait()
    t_out.join(); t_err.join()

    elapsed = time.time() - t0
    stdout = "".join(stdout_lines)

    # Parse peak_z lines: "  [seq_NNNN] frames=500  peak_z=0.183"
    # or "  [seq_NNNN] frames=500  LIFTED"  (when peak_z >= lift_height in generator)
    # Exclude padded sequences (seq_pad).
    peak_z_list = []
    for line in stdout.splitlines():
        if "seq_pad" in line:
            continue
        m = re.search(r"peak_z=([\d.]+)", line)
        if m:
            peak_z_list.append(float(m.group(1)))
        elif "LIFTED" in line and re.search(r"seq_\d+", line):
            # LIFTED means peak_z >= DEFAULT_CONFIG["lift_height"] = 0.20m
            peak_z_list.append(0.25)  # record as >0.20 sentinel

    hit = sum(1 for z in peak_z_list if z >= _LIFT_HEIGHT)

    print(f"\n  → peak_z: {[round(z, 3) for z in peak_z_list]}"
          f"  hit≥{_LIFT_HEIGHT}m: {hit}/{len(peak_z_list)}"
          f"  elapsed={elapsed:.0f}s")

    if proc.returncode != 0:
        print(f"  [ERROR] exit={proc.returncode}")

    return {
        "label":   probe.label,
        "probe":   probe,
        "peak_z":  peak_z_list,
        "hit":     hit,
        "elapsed": elapsed,
        "stdout":  stdout,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Physics material space sweep for VBD grasping.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--probes",  type=str, default=None,
                        help="Comma-separated probe labels to run (e.g. A,B,D). Default: all.")
    args = parser.parse_args()

    probes = SWEEPS
    if args.probes:
        wanted = set(p.strip().upper() for p in args.probes.split(","))
        probes = [p for p in SWEEPS if p.label.split(":")[0].upper() in wanted]
        if not probes:
            print(f"[sweep] No matching probes for {args.probes}. Available: {[p.label for p in SWEEPS]}")
            return

    results = []
    for probe in probes:
        r = run_probe(probe, dry_run=args.dry_run)
        results.append(r)

    # Summary table
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY  (lift_height = {:.2f}m)".format(_LIFT_HEIGHT))
    print(f"{'Probe':<22} {'hit/N':>6}  {'mean_z':>8}  {'max_z':>8}  {'mu_eff':>8}  {'F_n/p':>8}  {'Ffinger':>8}")
    print("-" * 70)
    import math
    for r in results:
        pz   = r["peak_z"]
        n    = len(pz)
        mean = np.mean(pz) if pz else 0.0
        mx   = np.max(pz)  if pz else 0.0
        p    = r["probe"]
        mue  = math.sqrt(p.soft_mu * p.contact_mu)
        fn   = p.soft_ke * PARTICLE_RADIUS
        print(f"  {r['label']:<20} {r['hit']}/{n}  {mean:>8.3f}  {mx:>8.3f}  "
              f"{mue:>8.2f}  {fn:>8.0f}N  {p.finger_effort:>6.0f}N")

    # Recommend best probe
    best = max(results, key=lambda r: (r["hit"], np.mean(r["peak_z"]) if r["peak_z"] else 0))
    print(f"\nBest probe: {best['label']}"
          f"  (hit={best['hit']}/{len(best['peak_z'])},"
          f" mean_z={np.mean(best['peak_z']):.3f}m)" if best["peak_z"] else "")
    print()
    print("To run the full pipeline with the best parameters, use:")
    if best["probe"]:
        p = best["probe"]
        print(f"  python scripts/generate_sequences.py --num_sequences 100 --num-worlds 4 \\")
        print(f"    --soft-ke={p.soft_ke} --soft-mu={p.soft_mu} --contact-mu={p.contact_mu} \\")
        print(f"    --density={p.density} --finger-ke={p.finger_ke} --finger-effort={p.finger_effort}")


if __name__ == "__main__":
    main()
