"""Tool 3 — Results Analyzer.

Reads the replay output JSON (Tool 2 output) and produces:

  1. Summary table  — per label: count, mean/std/min/max episode reward
  2. Confusion matrix — expected_high_reward vs actual_high_reward
  3. Histograms      — episode total reward distribution per label type
  4. Per-term curves — mean reward per term over time, one panel per label type
  5. Correlation     — scatter plot of expected vs actual reward

All figures are saved as PNG and optionally displayed interactively.

Usage
-----
python scripts/validation/franka_cube_pick/analyze_results.py \\
    --input  data/validation/replay.json \\
    --output data/validation/report/ \\
    [--show]   # display plots interactively

No Isaac Sim required — pure Python/matplotlib.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_TASK_ROOT   = Path(__file__).parent.parent
_OUTPUTS_DIR = _TASK_ROOT / "data" / "validation"

# ---- args ------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Analyze Franka cube pick replay results.")
parser.add_argument("--input",  type=str, default=str(_OUTPUTS_DIR / "replay.json"),
                    help="Path to replay output JSON.")
parser.add_argument("--output", type=str, default=str(_OUTPUTS_DIR / "report"),
                    help="Directory for output PNGs + report.")
parser.add_argument("--show",   action="store_true", help="Show plots interactively.")
parser.add_argument(
    "--high_reward_threshold", type=float, default=100.0,
    help="Episode total reward above this threshold is classified as HIGH (default: 100.0). "
         "Physics-based sequences: reachable_success ~1500, failures ~0-50 → 100 is safe.",
)
args = parser.parse_args()

# ---- helpers ---------------------------------------------------------------

LABEL_COLORS = {
    "reachable_success":   "#2ecc71",
    "reachable_failure":   "#e74c3c",
    "unreachable_success": "#3498db",
    "unreachable_failure": "#e67e22",
}

TERM_NAMES = [
    "approach_cube_reachable",
    "lift_cube_reachable",
    "cube_at_success_position",
    "go_to_signal_position",
    "action_rate",
    "joint_vel",
]

REWARD_WEIGHTS = {
    "approach_cube_reachable":   1.0,
    "lift_cube_reachable":      10.0,
    "cube_at_success_position": 15.0,
    "go_to_signal_position":    10.0,
    "action_rate":              -1e-4,
    "joint_vel":                -1e-4,
}


def label_str(label: dict) -> str:
    r = "reachable"   if label["reachable"] else "unreachable"
    s = "success"     if label["success"]   else "failure"
    return f"{r}_{s}"


def load_replay(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_fig(fig, name: str, out_dir: Path):
    p = out_dir / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"[analyze] Saved {p}")


# ---- main ------------------------------------------------------------------

def main():
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_replay(args.input)
    seqs = data["sequences"]
    thresh = args.high_reward_threshold

    # ---- Collect per-sequence data -----------------------------------------

    # episode_total[label_str] = list of float
    episode_totals = defaultdict(list)
    # episode_terms[label_str][term] = list of float
    episode_terms  = defaultdict(lambda: defaultdict(list))
    # per-frame curves[label_str][term] = list of (N_frames, N_seqs) arrays
    frame_curves   = defaultdict(lambda: defaultdict(list))

    confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}  # expected HIGH vs actual HIGH

    for seq in seqs:
        lbl    = label_str(seq["label"])
        ep_rew = seq["episode_rewards"]
        total  = ep_rew["total"]

        episode_totals[lbl].append(total)
        for term in TERM_NAMES:
            weighted = ep_rew.get(term, 0.0) * REWARD_WEIGHTS[term]
            episode_terms[lbl][term].append(weighted)

        # Per-frame curves (weighted)
        frame_totals_seq = []
        for frame in seq["frames"]:
            for term in TERM_NAMES:
                frame_curves[lbl][term].append(
                    frame["rewards"].get(term, 0.0) * REWARD_WEIGHTS[term]
                )
            frame_totals_seq.append(frame["rewards"].get("total", 0.0))

        # Confusion — use task completion (success_frame >= 0), not reward threshold.
        # Threshold-based classification is fragile: near-cube failure modes accumulate
        # approach reward that can exceed any fixed threshold. success_frame is set by
        # physics directly (cube_z > lift_height for reachable; ee near signal for unreachable).
        expected_high = seq["expected_high_reward"]
        actual_high   = seq.get("success_frame", -1) >= 0
        if expected_high and actual_high:      confusion["TP"] += 1
        elif expected_high and not actual_high: confusion["FN"] += 1
        elif not expected_high and actual_high: confusion["FP"] += 1
        else:                                   confusion["TN"] += 1

    # ---- 1. Summary table --------------------------------------------------

    print("\n" + "=" * 80)
    print("SUMMARY TABLE — Episode Total Reward by Label")
    print("=" * 80)
    print(f"{'Label':<28} {'N':>5} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 80)
    label_order = ["reachable_success", "reachable_failure", "unreachable_success", "unreachable_failure"]
    report_lines = [f"{'Label':<28} {'N':>5} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}\n" + "-" * 80]
    for lbl in label_order:
        vals = episode_totals.get(lbl, [])
        if not vals:
            continue
        arr = np.array(vals)
        row = f"{lbl:<28} {len(arr):>5} {arr.mean():>8.2f} {arr.std():>8.2f} {arr.min():>8.2f} {arr.max():>8.2f}"
        print(row)
        report_lines.append(row)
    print("=" * 80)

    print("\nCONFUSION MATRIX (criterion: success_frame >= 0, no threshold)")
    print(f"  True  Positives (expected SUCCESS, got SUCCESS): {confusion['TP']}")
    print(f"  False Negatives (expected SUCCESS, got FAILURE): {confusion['FN']}")
    print(f"  True  Negatives (expected FAILURE, got FAILURE): {confusion['TN']}")
    print(f"  False Positives (expected FAILURE, got SUCCESS): {confusion['FP']}")
    total_seqs = sum(confusion.values())
    accuracy = (confusion["TP"] + confusion["TN"]) / max(total_seqs, 1) * 100
    print(f"  Accuracy: {accuracy:.1f}%  ({total_seqs} sequences)")
    print()

    # ---- 2. Histograms — episode total reward per label --------------------

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Episode Total Reward Distribution by Label", fontsize=14)
    for ax, lbl in zip(axes.flat, label_order):
        vals = episode_totals.get(lbl, [])
        color = LABEL_COLORS.get(lbl, "gray")
        if vals:
            ax.hist(vals, bins=20, color=color, edgecolor="white", alpha=0.85)
            ax.axvline(np.mean(vals), color="black", linestyle="--", linewidth=1.5, label=f"mean={np.mean(vals):.1f}")
            ax.axvline(thresh, color="red", linestyle=":", linewidth=1.5, label=f"threshold={thresh:.0f}")
            ax.legend(fontsize=8)
        ax.set_title(lbl.replace("_", " "), fontsize=10)
        ax.set_xlabel("Episode Total Reward")
        ax.set_ylabel("Count")
    plt.tight_layout()
    save_fig(fig, "01_reward_histograms", out_dir)

    # ---- 3. Per-term episode reward stacked bar per label ------------------

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(label_order))
    width = 0.12
    for i, term in enumerate(TERM_NAMES):
        means = []
        for lbl in label_order:
            vals = episode_terms.get(lbl, {}).get(term, [0.0])
            means.append(np.mean(vals) if vals else 0.0)
        offset = (i - len(TERM_NAMES) / 2) * width
        ax.bar(x + offset, means, width=width, label=term, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in label_order], fontsize=9)
    ax.set_ylabel("Mean Weighted Episode Reward")
    ax.set_title("Per-Term Weighted Episode Reward by Label")
    ax.legend(fontsize=7, loc="upper right")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    save_fig(fig, "02_per_term_episode_reward", out_dir)

    # ---- 4. Per-frame reward curves per label ------------------------------

    # For each label, collect all frame rewards per term → compute mean over sequences
    # frame_curves[lbl][term] is a flat list — we need to reshape per-sequence

    seq_by_label = defaultdict(list)
    for seq in seqs:
        seq_by_label[label_str(seq["label"])].append(seq)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mean Per-Frame Reward Curves by Label (weighted)", fontsize=13)

    for ax, lbl in zip(axes.flat, label_order):
        seqs_lbl = seq_by_label.get(lbl, [])
        if not seqs_lbl:
            ax.set_title(f"{lbl} (no data)")
            continue

        min_len = min(len(s["frames"]) for s in seqs_lbl)
        times = [f["t"] for f in seqs_lbl[0]["frames"][:min_len]]

        for term in TERM_NAMES:
            term_curves = []
            for seq in seqs_lbl:
                curve = [f["rewards"].get(term, 0.0) * REWARD_WEIGHTS[term]
                         for f in seq["frames"][:min_len]]
                term_curves.append(curve)
            mean_curve = np.mean(term_curves, axis=0)
            ax.plot(times, mean_curve, label=term, linewidth=1.2)

        ax.set_title(lbl.replace("_", " "), fontsize=10)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("Weighted Reward")
        ax.legend(fontsize=6)
        ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    save_fig(fig, "03_per_frame_reward_curves", out_dir)

    # ---- 5. Correlation: expected vs actual ---------------------------------

    expected_vals = []
    actual_vals   = []
    colors_scatter = []
    for seq in seqs:
        lbl = label_str(seq["label"])
        expected_vals.append(1.0 if seq["expected_high_reward"] else 0.0)
        actual_vals.append(seq["episode_rewards"]["total"])
        colors_scatter.append(LABEL_COLORS.get(lbl, "gray"))

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(expected_vals, actual_vals, c=colors_scatter, alpha=0.6, s=30)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Expected LOW (failure)", "Expected HIGH (success)"])
    ax.set_ylabel("Actual Episode Total Reward")
    ax.set_title("Expected vs Actual Episode Reward")
    ax.axhline(thresh, color="red", linestyle="--", linewidth=1.5, label=f"threshold={thresh:.0f}")
    ax.legend()

    # Legend for label types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=LABEL_COLORS[l], label=l.replace("_", " ")) for l in label_order]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color="red", linestyle="--", label=f"threshold={thresh:.0f}")
    ], fontsize=8)

    plt.tight_layout()
    save_fig(fig, "04_expected_vs_actual_correlation", out_dir)

    # ---- 6. Reachability mask consistency check ---------------------------

    # For each frame, check that reachable_mask matches label
    mask_mismatch_count = defaultdict(int)
    mask_total_count    = defaultdict(int)
    for seq in seqs:
        lbl = label_str(seq["label"])
        expected_mask = 1.0 if seq["label"]["reachable"] else 0.0
        for frame in seq["frames"]:
            mask_total_count[lbl] += 1
            if abs(frame["reachable_mask"] - expected_mask) > 0.5:
                mask_mismatch_count[lbl] += 1

    print("REACHABILITY MASK CONSISTENCY")
    print("-" * 50)
    mask_report_lines = []
    for lbl in label_order:
        total = mask_total_count.get(lbl, 0)
        mismatches = mask_mismatch_count.get(lbl, 0)
        if total > 0:
            pct = mismatches / total * 100
            status = "OK" if pct < 1.0 else "WARNING"
            line = f"  {lbl:<28}: {mismatches}/{total} mismatches ({pct:.1f}%) [{status}]"
            print(line)
            mask_report_lines.append(line)
    print()

    # ---- 7. Success frame distribution + variance -------------------------

    # Collect success_frame and joint_pos_variance_mean per label
    success_frames   = defaultdict(list)   # label → list of frame indices (-1 = never)
    variance_by_lbl  = defaultdict(list)   # label → list of variance values

    for seq in seqs:
        lbl = label_str(seq["label"])
        sf  = seq.get("success_frame", -1)
        var = seq.get("joint_pos_variance_mean", 0.0)
        success_frames[lbl].append(sf)
        variance_by_lbl[lbl].append(var)

    # Success frame distribution (successful sequences only — sf >= 0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Success Frame Distribution (frame index when success first detected)", fontsize=13)
    for ax, lbl in zip(axes.flat, label_order):
        frames_list = success_frames.get(lbl, [])
        hit   = [f for f in frames_list if f >= 0]
        miss  = [f for f in frames_list if f < 0]
        color = LABEL_COLORS.get(lbl, "gray")
        ax.set_title(lbl.replace("_", " "), fontsize=10)
        if hit:
            ax.hist(hit, bins=20, color=color, edgecolor="white", alpha=0.85, label=f"success (n={len(hit)})")
            ax.axvline(np.mean(hit), color="black", linestyle="--", linewidth=1.5,
                       label=f"mean={np.mean(hit):.0f}")
        ax.set_xlabel("Success Frame Index")
        ax.set_ylabel("Count")
        n_total = len(frames_list)
        n_hit   = len(hit)
        ax.set_title(f"{lbl.replace('_', ' ')}\n({n_hit}/{n_total} hit success)", fontsize=9)
        ax.legend(fontsize=7)
    plt.tight_layout()
    save_fig(fig, "05_success_frame_distribution", out_dir)

    # Variance distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, lbl in enumerate(label_order):
        vals = variance_by_lbl.get(lbl, [])
        if vals:
            color = LABEL_COLORS.get(lbl, "gray")
            ax.hist(vals, bins=20, color=color, edgecolor="white", alpha=0.6,
                    label=f"{lbl.replace('_', ' ')} (μ={np.mean(vals):.4f})")
    ax.set_xlabel("joint_pos_variance_mean (mean |gen − replay| joint pos, rad)")
    ax.set_ylabel("Count")
    ax.set_title("Physics Non-Determinism: Generator vs Replay Joint Position Variance")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, "06_joint_pos_variance", out_dir)

    # Print variance summary
    print("PHYSICS VARIANCE (mean |gen_joint_pos − replay_joint_pos|)")
    print("-" * 50)
    variance_report_lines = []
    for lbl in label_order:
        vals = variance_by_lbl.get(lbl, [])
        if vals:
            arr = np.array(vals)
            line = f"  {lbl:<28}: mean={arr.mean():.4f}  std={arr.std():.4f}  max={arr.max():.4f}"
            print(line)
            variance_report_lines.append(line)
    print()

    # Print success frame summary
    print("SUCCESS FRAME SUMMARY")
    print("-" * 50)
    sf_report_lines = []
    for lbl in label_order:
        frames_list = success_frames.get(lbl, [])
        if not frames_list:
            continue
        hit = [f for f in frames_list if f >= 0]
        n = len(frames_list)
        if hit:
            arr = np.array(hit)
            line = f"  {lbl:<28}: {len(hit)}/{n} hit  mean_frame={arr.mean():.0f}  min={arr.min()}  max={arr.max()}"
        else:
            line = f"  {lbl:<28}: 0/{n} hit (none reached success)"
        print(line)
        sf_report_lines.append(line)
    print()

    # ---- Save text report --------------------------------------------------
    report_path = out_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write("Franka Cube Pick Validation Report\n")
        f.write(f"Source: {args.input}\n")
        f.write(f"Threshold (high/low): {thresh}\n\n")
        f.write(f"Total sequences: {len(seqs)}\n\n")
        f.write("CONFUSION MATRIX\n")
        f.write(f"  TP={confusion['TP']}  FN={confusion['FN']}  TN={confusion['TN']}  FP={confusion['FP']}\n")
        f.write(f"  Accuracy: {accuracy:.1f}%\n\n")
        for line in report_lines:
            f.write(line + "\n")
        f.write("\nREACHABILITY MASK CONSISTENCY\n")
        f.write("-" * 50 + "\n")
        for line in mask_report_lines:
            f.write(line + "\n")
        f.write("\nSUCCESS FRAME SUMMARY\n")
        f.write("-" * 50 + "\n")
        for line in sf_report_lines:
            f.write(line + "\n")
        f.write("\nPHYSICS VARIANCE\n")
        f.write("-" * 50 + "\n")
        for line in variance_report_lines:
            f.write(line + "\n")
    print(f"[analyze] Report saved → {report_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
