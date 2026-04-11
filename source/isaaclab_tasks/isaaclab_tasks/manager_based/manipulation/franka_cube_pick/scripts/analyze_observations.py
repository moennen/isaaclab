"""Tool 5 — Observation Analyzer.

Two input modes:

  Mode A — Full 47-dim RL obs (recommended):
    --obs    reference_ik_observations.json  (Tool 6 output)
    --replay reference_ik_replay.json        (Tool 2 output, for reward data)

  Mode B — Reconstructed 25-dim obs (fallback, no Tool 6 needed):
    --seqs   reference_ik_sequences.json     (Tool 1 output)
    --replay reference_ik_replay.json        (Tool 2 output)

Produces 7 analyses:
  1. Obs–Reward Pearson correlation heatmap
  2. Obs–Action Pearson correlation heatmap
  3. Markov property test — ΔR² per obs dim when adding obs[t-1]
  4. PCA redundancy — cumulative explained variance
  5. Label discriminability — Fisher ratio (reachable vs unreachable, success vs failure)
  6. Reward predictability — R² of linear obs → per-frame reward per label group
  7. Temporal autocorrelation — lag 1–10 per obs dim

No Isaac Sim required — pure Python/numpy/sklearn/matplotlib.

Usage
-----
# Full 47-dim analysis (after running compute_observations.py):
python scripts/analyze_observations.py \\
    --obs    data/validation/reference_ik_observations.json \\
    --replay data/validation/reference_ik_replay.json \\
    --output data/validation/reference_ik_obs_report/

# Fallback 25-dim analysis (sequences + replay only):
python scripts/analyze_observations.py \\
    --seqs   data/validation/reference_ik_sequences.json \\
    --replay data/validation/reference_ik_replay.json \\
    --output data/validation/reference_ik_obs_report/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

_TASK_ROOT   = Path(__file__).parent.parent
_OUTPUTS_DIR = _TASK_ROOT / "data" / "validation"

# ---- CLI -------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Analyze Franka cube pick observations.")
parser.add_argument(
    "--obs", type=str, default=None,
    help="[Mode A] Path to reference_ik_observations.json (Tool 6 output). "
         "When provided, uses full 47-dim RL obs vector.",
)
parser.add_argument(
    "--seqs", type=str,
    default=str(_OUTPUTS_DIR / "reference_ik_sequences.json"),
    help="[Mode B] Path to sequences JSON (Tool 1 output). Used when --obs is absent.",
)
parser.add_argument(
    "--replay", type=str,
    default=str(_OUTPUTS_DIR / "reference_ik_replay.json"),
    help="Path to replay JSON (Tool 2 output). Required for reward data in both modes.",
)
parser.add_argument(
    "--output", type=str,
    default=str(_OUTPUTS_DIR / "reference_ik_obs_report"),
    help="Output directory for PNGs + report.txt.",
)
parser.add_argument("--show", action="store_true", help="Show plots interactively.")
args = parser.parse_args()

# ---- Fallback observation layout (Mode B, 25 dims) -------------------------

_OBS_NAMES_25 = [
    "cube_x", "cube_y", "cube_z",
    "ee_x", "ee_y", "ee_z",
    "j0", "j1", "j2", "j3", "j4", "j5", "j6",
    "f0", "f1",
    "jv0", "jv1", "jv2", "jv3", "jv4", "jv5", "jv6",
    "fv0", "fv1",
    "grip",
]  # 25 total

ACTION_NAMES = [f"cmd_j{i}" for i in range(7)] + ["cmd_f0", "cmd_f1"]  # 9 total

# Populated by main() from either the --obs file (47 dims) or _OBS_NAMES_25 (25 dims)
OBS_NAMES: list = []

REWARD_TERMS = [
    "approach_cube_reachable",
    "grip_cube_reachable",
    "lift_cube_reachable",
    "cube_at_success_position",
    "go_to_signal_position",
    "signal_reached_unreachable",
    "action_rate",
    "joint_vel",
]

REWARD_WEIGHTS = {
    "approach_cube_reachable":    1.0,
    "grip_cube_reachable":        5.0,
    "lift_cube_reachable":       10.0,
    "cube_at_success_position":  15.0,
    "go_to_signal_position":      1.0,
    "signal_reached_unreachable": 10.0,
    "action_rate":               -1e-4,
    "joint_vel":                 -1e-4,
}

LABEL_COLORS = {
    "reachable_success":   "#2ecc71",
    "reachable_failure":   "#e74c3c",
    "unreachable_success": "#3498db",
    "unreachable_failure": "#e67e22",
}


# ---- Helpers ---------------------------------------------------------------

def label_str(label: dict) -> str:
    r = "reachable"   if label["reachable"] else "unreachable"
    s = "success"     if label["success"]   else "failure"
    return f"{r}_{s}"


def save_fig(fig, name: str, out_dir: Path, show: bool = False):
    p = out_dir / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"[obs] Saved {p}")
    if show:
        plt.show()
    plt.close(fig)


# ---- Data loading ----------------------------------------------------------

def _load_rewards_from_replay(replay_path: str, seq_ids_by_seqname: dict):
    """Load per-frame weighted rewards aligned to the sequences in seq_ids_by_seqname.

    Returns parallel lists (same frame order as the caller's seq iteration):
        rew_dict  : dict[term] → list of floats
        tot_list  : list of floats
        mask_list : list of bools
    """
    with open(replay_path) as f:
        replay_data = json.load(f)
    rseqs = {r["id"]: r for r in replay_data["sequences"]}

    rew_dict  = {t: [] for t in REWARD_TERMS}
    tot_list  = []
    mask_list = []

    for seq_name, n_frames in seq_ids_by_seqname.items():
        rseq    = rseqs[seq_name]
        rframes = rseq["frames"]
        assert len(rframes) == n_frames, f"Frame count mismatch for {seq_name}"
        for rf in rframes:
            frame_total = 0.0
            for term in REWARD_TERMS:
                raw      = rf["rewards"].get(term, 0.0)
                weighted = raw * REWARD_WEIGHTS[term]
                rew_dict[term].append(weighted)
                frame_total += weighted
            tot_list.append(frame_total)
            mask_list.append(rf["reachable_mask"] > 0.5)

    return rew_dict, tot_list, mask_list


def load_data_from_obs(obs_path: str, replay_path: str):
    """Mode A — full 47-dim obs from compute_observations.py output.

    Returns
    -------
    obs         : (N, 47) float32
    obs_names   : list[str]  — 47 dim names
    actions     : (N, 9)  float32  — last 9 dims of obs (last_action)
    rewards     : dict[term] → (N,) float32
    total       : (N,)  float32
    mask        : (N,)  bool
    labels      : (N,)  str
    seqids      : (N,)  int
    """
    with open(obs_path) as f:
        obs_data = json.load(f)

    obs_names = obs_data["obs_names"]      # 47 strings
    seqs      = obs_data["sequences"]

    obs_list  = []
    lbl_list  = []
    sid_list  = []
    seq_ids_by_seqname = {}   # {seq_id: n_frames}

    for seq_idx, seq in enumerate(seqs):
        lbl = label_str(seq["label"])
        frames = seq["frames"]
        seq_ids_by_seqname[seq["id"]] = len(frames)
        for f in frames:
            obs_list.append(f["obs"])
            lbl_list.append(lbl)
            sid_list.append(seq_idx)

    obs    = np.array(obs_list, dtype=np.float32)
    labels = np.array(lbl_list)
    seqids = np.array(sid_list, dtype=np.int32)

    # Last 9 dims of obs are last_action
    action_start = len(obs_names) - len(ACTION_NAMES)
    actions = obs[:, action_start:]

    rew_dict, tot_list, mask_list = _load_rewards_from_replay(replay_path, seq_ids_by_seqname)

    rewards = {t: np.array(v, dtype=np.float32) for t, v in rew_dict.items()}
    total   = np.array(tot_list,  dtype=np.float32)
    mask    = np.array(mask_list, dtype=bool)

    n_seqs, n_frames = len(seqs), len(seqs[0]["frames"])
    print(f"[obs] Mode A (47-dim): {len(obs):,} frames ({n_seqs} seqs × {n_frames} frames/seq)")
    return obs, obs_names, actions, rewards, total, mask, labels, seqids


def load_data_from_seqs(seqs_path: str, replay_path: str):
    """Mode B — reconstructed 25-dim obs from sequences + replay JSON.

    Returns same tuple as load_data_from_obs but obs is (N, 25) and
    obs_names = _OBS_NAMES_25.
    """
    with open(seqs_path) as f:
        seqs_data = json.load(f)
    seqs = seqs_data["sequences"]

    obs_list  = []
    act_list  = []
    lbl_list  = []
    sid_list  = []
    seq_ids_by_seqname = {}

    for seq_idx, seq in enumerate(seqs):
        lbl    = label_str(seq["label"])
        frames = seq["frames"]
        seq_ids_by_seqname[seq["id"]] = len(frames)
        for f in frames:
            jp = f["joint_pos"]
            jv = f["joint_vel"]
            jc = f["joint_pos_cmd"]
            ep = f["ee_pos_w"]
            rp = f["robot_pos_w"]
            cp = f["cube_pos_w"]
            cube_b = [cp[0]-rp[0], cp[1]-rp[1], cp[2]-rp[2]]
            ee_b   = [ep[0]-rp[0], ep[1]-rp[1], ep[2]-rp[2]]
            grip   = 1.0 if f["gripper_closed"] else 0.0
            obs_list.append(cube_b + ee_b + jp[:7] + jp[7:9] + jv[:7] + jv[7:9] + [grip])
            act_list.append(jc)
            lbl_list.append(lbl)
            sid_list.append(seq_idx)

    obs     = np.array(obs_list, dtype=np.float32)
    actions = np.array(act_list, dtype=np.float32)
    labels  = np.array(lbl_list)
    seqids  = np.array(sid_list, dtype=np.int32)

    rew_dict, tot_list, mask_list = _load_rewards_from_replay(replay_path, seq_ids_by_seqname)
    rewards = {t: np.array(v, dtype=np.float32) for t, v in rew_dict.items()}
    total   = np.array(tot_list,  dtype=np.float32)
    mask    = np.array(mask_list, dtype=bool)

    n_seqs, n_frames = len(seqs), len(seqs[0]["frames"])
    print(f"[obs] Mode B (25-dim): {len(obs):,} frames ({n_seqs} seqs × {n_frames} frames/seq)")
    return obs, _OBS_NAMES_25, actions, rewards, total, mask, labels, seqids


# ---- Analysis 1: Obs–Reward Pearson correlation ----------------------------

def analysis_obs_reward_correlation(obs, rewards, out_dir, show):
    n_obs = obs.shape[1]
    n_rew = len(REWARD_TERMS)
    corr  = np.zeros((n_obs, n_rew), dtype=np.float32)

    for j, term in enumerate(REWARD_TERMS):
        r = rewards[term]
        for i in range(n_obs):
            o = obs[:, i]
            if o.std() < 1e-8 or r.std() < 1e-8:
                corr[i, j] = 0.0
            else:
                corr[i, j] = float(np.corrcoef(o, r)[0, 1])

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_rew))
    ax.set_xticklabels([t.replace("_", "\n") for t in REWARD_TERMS], fontsize=7)
    ax.set_yticks(range(n_obs))
    ax.set_yticklabels(OBS_NAMES, fontsize=8)
    ax.set_title("Analysis 1: Obs–Reward Pearson Correlation", fontsize=12)
    plt.colorbar(im, ax=ax, label="Pearson r")
    for i in range(n_obs):
        for j in range(n_rew):
            v = corr[i, j]
            if abs(v) > 0.15:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > 0.5 else "black")
    plt.tight_layout()
    save_fig(fig, "01_obs_reward_correlation", out_dir, show)
    return corr


# ---- Analysis 2: Obs–Action Pearson correlation ----------------------------

def analysis_obs_action_correlation(obs, actions, out_dir, show):
    n_obs = obs.shape[1]
    n_act = actions.shape[1]
    corr  = np.zeros((n_obs, n_act), dtype=np.float32)

    for j in range(n_act):
        a = actions[:, j]
        for i in range(n_obs):
            o = obs[:, i]
            if o.std() < 1e-8 or a.std() < 1e-8:
                corr[i, j] = 0.0
            else:
                corr[i, j] = float(np.corrcoef(o, a)[0, 1])

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_act))
    ax.set_xticklabels(ACTION_NAMES, fontsize=8)
    ax.set_yticks(range(n_obs))
    ax.set_yticklabels(OBS_NAMES, fontsize=8)
    ax.set_title("Analysis 2: Obs–Action Pearson Correlation", fontsize=12)
    plt.colorbar(im, ax=ax, label="Pearson r")
    for i in range(n_obs):
        for j in range(n_act):
            v = corr[i, j]
            if abs(v) > 0.3:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > 0.5 else "black")
    plt.tight_layout()
    save_fig(fig, "02_obs_action_correlation", out_dir, show)
    return corr


# ---- Analysis 3: Markov property test --------------------------------------

def analysis_markov_test(obs, seqids, out_dir, show):
    """For each obs dim i, compare R² of predicting obs[t+1, i] from:
      - all 25 obs dims at t alone              → R²_1
      - all 25 obs dims at [t-1, t] (50 dims)   → R²_2

    ΔR² = R²_2 − R²_1 measures how much past state helps predict the next.
    ΔR² > 0.05 indicates the observation is not fully Markovian for that dim.

    Uses Ridge regression (α=1.0) to handle multicollinearity between t and t-1.
    In-sample R² is used; values are diagnostic, not predictive.
    """
    n_obs = obs.shape[1]
    n_seqs = int(seqids.max()) + 1

    # Build valid triplets (t-1, t, t+1) within the same sequence
    valid = []
    for i in range(1, len(obs) - 1):
        if seqids[i - 1] == seqids[i] == seqids[i + 1]:
            valid.append(i)
    valid = np.array(valid)

    X1 = obs[valid]                                      # (M, 25)  — obs[t]
    X2 = np.hstack([obs[valid - 1], obs[valid]])         # (M, 50)  — [obs[t-1], obs[t]]
    Y  = obs[valid + 1]                                  # (M, 25)  — obs[t+1]

    sc1 = StandardScaler().fit(X1)
    sc2 = StandardScaler().fit(X2)
    X1s = sc1.transform(X1)
    X2s = sc2.transform(X2)

    r2_1 = np.zeros(n_obs, dtype=np.float32)
    r2_2 = np.zeros(n_obs, dtype=np.float32)

    for i in range(n_obs):
        y = Y[:, i]
        if y.std() < 1e-8:
            r2_1[i] = 1.0
            r2_2[i] = 1.0
            continue
        r2_1[i] = max(0.0, float(r2_score(y, Ridge(alpha=1.0).fit(X1s, y).predict(X1s))))
        r2_2[i] = max(0.0, float(r2_score(y, Ridge(alpha=1.0).fit(X2s, y).predict(X2s))))

    delta_r2 = r2_2 - r2_1

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    x = np.arange(n_obs)

    ax = axes[0]
    ax.bar(x, r2_1, label="R² from obs[t] only", alpha=0.75, color="#3498db")
    ax.bar(x, delta_r2.clip(0), bottom=r2_1, label="ΔR² (gain from obs[t-1])",
           alpha=0.75, color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(OBS_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("R²")
    ax.set_title("Analysis 3: Markov Test — predicting obs[t+1] from obs[t] vs [obs[t-1], obs[t]]")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    colors = ["#e74c3c" if d > 0.05 else "#2ecc71" for d in delta_r2]
    ax.bar(x, delta_r2.clip(0), color=colors, alpha=0.85)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.2, label="ΔR² = 0.05 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(OBS_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("ΔR²")
    ax.set_title("ΔR² per dim (red = history adds info, ΔR² > 0.05)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig, "03_markov_test", out_dir, show)
    return delta_r2, r2_1, r2_2


# ---- Analysis 4: PCA Redundancy --------------------------------------------

def analysis_pca_redundancy(obs, out_dir, show):
    """How many PCs are needed to explain 90/95/99% of obs variance?"""
    obs_scaled = StandardScaler().fit_transform(obs)
    pca        = PCA().fit(obs_scaled)
    cumvar     = np.cumsum(pca.explained_variance_ratio_)

    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    n99 = int(np.searchsorted(cumvar, 0.99)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(range(1, len(cumvar) + 1), cumvar, "b-o", markersize=4)
    ax.axhline(0.90, color="orange",  linestyle="--", label=f"90% → {n90} PCs")
    ax.axhline(0.95, color="red",     linestyle="--", label=f"95% → {n95} PCs")
    ax.axhline(0.99, color="purple",  linestyle="--", label=f"99% → {n99} PCs")
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Analysis 4a: PCA Cumulative Explained Variance")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, len(cumvar))

    ax = axes[1]
    ax.bar(range(1, len(pca.explained_variance_ratio_) + 1),
           pca.explained_variance_ratio_, color="#3498db", alpha=0.85)
    ax.set_xlabel("PC Index")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Analysis 4b: Per-Component Explained Variance")

    plt.tight_layout()
    save_fig(fig, "04_pca_redundancy", out_dir, show)
    return n90, n95, n99, pca.explained_variance_ratio_


# ---- Analysis 5: Label Discriminability (Fisher ratio) ---------------------

def analysis_label_discriminability(obs, mask, labels, out_dir, show):
    """Fisher discriminant ratio F = (μ_a − μ_b)² / (σ_a² + σ_b² + ε) per dim.

    High F → dimension clearly separates the two groups → useful for policy.

    Two splits:
      (a) reachable vs unreachable  (determined by per-frame reachable_mask)
      (b) success vs failure        (determined by sequence-level label)
    """
    def fisher(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
        var_a, var_b = a.var(axis=0), b.var(axis=0)
        return (mu_a - mu_b) ** 2 / (var_a + var_b + 1e-8)

    # (a) reachable vs unreachable
    f_reach = fisher(obs[mask], obs[~mask])

    # (b) success vs failure
    success_mask = np.array(["success" in l for l in labels])
    f_succ = fisher(obs[success_mask], obs[~success_mask])

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    x = np.arange(len(OBS_NAMES))

    for ax, f_vals, title in zip(axes,
                                  [f_reach, f_succ],
                                  ["Reachable vs Unreachable", "Success vs Failure"]):
        colors = ["#e74c3c" if v > 1.0 else "#3498db" for v in f_vals]
        ax.bar(x, f_vals, color=colors, alpha=0.85)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="F = 1.0")
        ax.set_xticks(x)
        ax.set_xticklabels(OBS_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Fisher Discriminant Ratio")
        ax.set_title(f"Analysis 5: Label Discriminability — {title}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig, "05_label_discriminability", out_dir, show)
    return f_reach, f_succ


# ---- Analysis 6: Reward Predictability (R²) --------------------------------

def analysis_reward_predictability(obs, total, labels, out_dir, show):
    """R² of Ridge regression obs → per-frame total reward, per label group.

    High R² means current observation is a good linear predictor of immediate reward.
    Computed within each label group (same branch, same outcome) to remove branch
    confounding.
    """
    label_order = ["reachable_success", "reachable_failure",
                   "unreachable_success", "unreachable_failure"]
    r2_per_label = {}
    sc = StandardScaler()

    for lbl in label_order:
        idx = labels == lbl
        if idx.sum() < 20:
            r2_per_label[lbl] = float("nan")
            continue
        X = obs[idx]
        y = total[idx]
        if y.std() < 1e-8:
            r2_per_label[lbl] = float("nan")
            continue
        X_s = sc.fit_transform(X)
        r2_per_label[lbl] = float(r2_score(y, Ridge(alpha=1.0).fit(X_s, y).predict(X_s)))

    # Global across all frames
    X_s_all = sc.fit_transform(obs)
    r2_global = float(r2_score(total, Ridge(alpha=1.0).fit(X_s_all, total).predict(X_s_all)))

    all_labels = label_order + ["ALL"]
    all_r2 = [r2_per_label.get(l, float("nan")) for l in label_order] + [r2_global]
    colors = [LABEL_COLORS.get(l, "gray") for l in label_order] + ["#555555"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(all_labels)), all_r2, color=colors, alpha=0.85)
    for bar, v in zip(bars, all_r2):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, max(v + 0.02, 0.02),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels([l.replace("_", "\n") for l in all_labels], fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("Analysis 6: Reward Predictability — R² of obs → per-frame reward")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    save_fig(fig, "06_reward_predictability", out_dir, show)
    return r2_per_label, r2_global


# ---- Analysis 7: Temporal Autocorrelation ----------------------------------

def analysis_temporal_autocorrelation(obs, seqids, out_dir, show):
    """Lag-1 through lag-10 autocorrelation per obs dim, averaged over sequences.

    High lag-1 autocorrelation: slowly changing dimension (may carry stale info).
    Low lag-1 autocorrelation: rapidly changing dimension (may be noisy).
    """
    n_obs  = obs.shape[1]
    n_seqs = int(seqids.max()) + 1
    max_lag = 10

    seq_frames = [np.where(seqids == i)[0] for i in range(n_seqs)]

    ac = np.zeros((n_obs, max_lag), dtype=np.float32)
    for i in range(n_obs):
        for lag in range(1, max_lag + 1):
            corrs = []
            for idxs in seq_frames:
                if len(idxs) <= lag:
                    continue
                x_t  = obs[idxs[:-lag], i]
                x_tp = obs[idxs[lag:], i]
                if x_t.std() < 1e-8 or x_tp.std() < 1e-8:
                    continue
                corrs.append(float(np.corrcoef(x_t, x_tp)[0, 1]))
            ac[i, lag - 1] = float(np.mean(corrs)) if corrs else 0.0

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(ac, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(max_lag))
    ax.set_xticklabels([f"lag {l+1}" for l in range(max_lag)], fontsize=9)
    ax.set_yticks(range(n_obs))
    ax.set_yticklabels(OBS_NAMES, fontsize=8)
    ax.set_title("Analysis 7: Temporal Autocorrelation per Obs Dim (avg over seqs)", fontsize=12)
    plt.colorbar(im, ax=ax, label="Pearson r")
    for i in range(n_obs):
        for j in range(max_lag):
            v = ac[i, j]
            if abs(v) > 0.5:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > 0.7 else "black")
    plt.tight_layout()
    save_fig(fig, "07_temporal_autocorrelation", out_dir, show)
    return ac


# ---- Report ----------------------------------------------------------------

def write_report(
    out_dir, n_frames,
    corr_rw, corr_act,
    delta_r2, r2_1, r2_2,
    n90, n95, n99,
    f_reach, f_succ,
    r2_per_label, r2_global,
    ac,
):
    lines = []
    lines.append("Franka Cube Pick — Observation Analysis Report")
    lines.append("=" * 80)
    lines.append(f"Observation vector : {len(OBS_NAMES)} dims ({', '.join(OBS_NAMES)})")
    lines.append(f"Total frames       : {n_frames:,}")
    lines.append("")

    # 1. Obs–Reward
    lines.append("1. OBS–REWARD CORRELATION (top |r| ≥ 0.20)")
    lines.append("-" * 60)
    rows = []
    for i, oname in enumerate(OBS_NAMES):
        for j, tname in enumerate(REWARD_TERMS):
            r = float(corr_rw[i, j])
            if abs(r) >= 0.20:
                rows.append((abs(r), oname, tname, r))
    rows.sort(reverse=True)
    for _, oname, tname, r in rows[:25]:
        lines.append(f"  {oname:<12} ↔ {tname:<32}  r = {r:+.3f}")
    lines.append("")

    # 2. Obs–Action
    lines.append("2. OBS–ACTION CORRELATION (top |r| ≥ 0.40)")
    lines.append("-" * 60)
    rows = []
    for i, oname in enumerate(OBS_NAMES):
        for j, aname in enumerate(ACTION_NAMES):
            r = float(corr_act[i, j])
            if abs(r) >= 0.40:
                rows.append((abs(r), oname, aname, r))
    rows.sort(reverse=True)
    for _, oname, aname, r in rows[:25]:
        lines.append(f"  {oname:<12} ↔ {aname:<14}  r = {r:+.3f}")
    if not rows:
        lines.append("  (none above threshold)")
    lines.append("")

    # 3. Markov test
    lines.append("3. MARKOV PROPERTY TEST (ΔR² > 0.05 = history helps)")
    lines.append("-" * 60)
    non_markov = sorted(
        [(OBS_NAMES[i], float(delta_r2[i]), float(r2_1[i]), float(r2_2[i]))
         for i in range(len(OBS_NAMES)) if float(delta_r2[i]) > 0.05],
        key=lambda x: -x[1],
    )
    if non_markov:
        for name, dr2, r1, r2 in non_markov:
            lines.append(f"  {name:<12}  ΔR²={dr2:.3f}  R²(t)={r1:.3f} → R²(t,t-1)={r2:.3f}")
    else:
        lines.append("  (none — observation vector appears Markovian at this resolution)")
    lines.append("")

    # 4. PCA
    lines.append("4. PCA REDUNDANCY")
    lines.append("-" * 60)
    lines.append(f"  90% variance → {n90} PCs  (of {len(OBS_NAMES)} dims, {n90/len(OBS_NAMES):.0%} of full dim)")
    lines.append(f"  95% variance → {n95} PCs")
    lines.append(f"  99% variance → {n99} PCs")
    lines.append("")

    # 5. Discriminability
    lines.append("5. LABEL DISCRIMINABILITY (top Fisher ratio per split)")
    lines.append("-" * 60)
    lines.append("  Reachable vs Unreachable (top 8):")
    for name, f in sorted(zip(OBS_NAMES, f_reach), key=lambda x: -x[1])[:8]:
        lines.append(f"    {name:<12}  F={f:.2f}")
    lines.append("  Success vs Failure (top 8):")
    for name, f in sorted(zip(OBS_NAMES, f_succ), key=lambda x: -x[1])[:8]:
        lines.append(f"    {name:<12}  F={f:.2f}")
    lines.append("")

    # 6. Reward predictability
    lines.append("6. REWARD PREDICTABILITY (R² of obs → per-frame reward)")
    lines.append("-" * 60)
    label_order = ["reachable_success", "reachable_failure",
                   "unreachable_success", "unreachable_failure"]
    for lbl in label_order:
        r2 = r2_per_label.get(lbl, float("nan"))
        lines.append(f"  {lbl:<28}: R² = {r2:.3f}")
    lines.append(f"  {'ALL':<28}: R² = {r2_global:.3f}")
    lines.append("")

    # 7. Autocorrelation
    lines.append("7. TEMPORAL AUTOCORRELATION (lag-1, avg over seqs)")
    lines.append("-" * 60)
    lag1 = ac[:, 0]
    fast = [(OBS_NAMES[i], float(lag1[i])) for i in range(len(OBS_NAMES)) if lag1[i] < 0.5]
    slow = [(OBS_NAMES[i], float(lag1[i])) for i in range(len(OBS_NAMES)) if lag1[i] >= 0.95]
    lines.append("  Fast-changing dims (lag-1 < 0.5 — high noise, little smoothness):")
    if fast:
        for name, v in sorted(fast, key=lambda x: x[1]):
            lines.append(f"    {name:<12}  ac₁={v:.3f}")
    else:
        lines.append("    (none)")
    lines.append("  Slow-changing dims (lag-1 ≥ 0.95 — near-constant per step):")
    if slow:
        for name, v in sorted(slow, key=lambda x: -x[1]):
            lines.append(f"    {name:<12}  ac₁={v:.3f}")
    else:
        lines.append("    (none)")
    lines.append("")

    # Recommendations
    lines.append("=" * 80)
    lines.append("RECOMMENDATIONS FOR OBSERVATION DESIGN")
    lines.append("=" * 80)
    lines.append("")
    lines.append("A. POTENTIALLY MISSING OBSERVATIONS")
    lines.append("-" * 60)
    lines.append("  • cube_vel_xyz (3 dims): cube velocity would improve Markov completeness")
    lines.append("    for post-contact dynamics. cube_z is tracked but not its velocity;")
    lines.append("    cube_vel_z would help the policy detect whether the cube is falling.")
    lines.append("")
    lines.append("  • dist_cube_xy (1 dim): horizontal distance from robot base.")
    lines.append("    Derivable from (cube_x, cube_y) but making it explicit reduces policy")
    lines.append("    burden for reachability classification.")
    lines.append("")
    lines.append("  • ee_to_cube_xyz (3 dims): vector from EE to cube = cube_pos - ee_pos.")
    lines.append("    This is the most direct input for approach_cube reward; high r expected.")
    lines.append("    Currently the policy must infer it from separate cube_pos and ee_pos.")
    lines.append("")
    lines.append("  • gripper_width (1 dim): continuous sum of finger joint positions")
    lines.append("    (= f0 + f1, range 0..0.08). Currently only binary grip is in obs.")
    lines.append("    Continuous width gives fine-grained grasp feedback unavailable from binary.")
    lines.append("")
    lines.append("B. POSSIBLY REDUNDANT OBSERVATIONS")
    lines.append("-" * 60)
    lines.append("  • If PCA 90% threshold uses ≤ 15 PCs (≤60% of dims), the joint space")
    lines.append("    has significant correlations that could be reduced.")
    lines.append("  • Slow-changing dims (lag-1 ≥ 0.98) add negligible new info per step;")
    lines.append("    frame-stacking 2 steps captures dynamics without increasing obs size.")
    lines.append("")
    lines.append("C. REWARD–OBSERVATION ALIGNMENT")
    lines.append("-" * 60)
    lines.append("  • Dims with low Fisher ratio for BOTH splits (reachable/success) carry")
    lines.append("    little discriminating information and may be prunable.")
    lines.append("  • Low R² in reward predictability (within a label group) suggests the")
    lines.append("    current obs cannot linearly predict the reward — either the reward is")
    lines.append("    delayed (sparse) or depends on missing derived features.")
    lines.append("")
    lines.append("D. MARKOV COMPLETENESS")
    lines.append("-" * 60)
    lines.append("  • Dims with ΔR² > 0.05 indicate that knowing obs[t-1] helps predict")
    lines.append("    obs[t+1] beyond what obs[t] alone provides. The most common cause is")
    lines.append("    missing velocity terms (position dims look non-Markovian when velocity")
    lines.append("    is absent). If arm velocities do NOT appear in the non-Markov list,")
    lines.append("    including joint_vel is sufficient for Markov completeness at this scale.")
    lines.append("")

    report_path = out_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[obs] Report saved → {report_path}")


# ---- Main ------------------------------------------------------------------

def main():
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.obs is not None:
        obs, obs_names, actions, rewards, total, mask, labels, seqids = \
            load_data_from_obs(args.obs, args.replay)
        print(f"[obs] Mode A — {len(obs_names)}-dim obs from {args.obs}")
    else:
        obs, obs_names, actions, rewards, total, mask, labels, seqids = \
            load_data_from_seqs(args.seqs, args.replay)
        print(f"[obs] Mode B — {len(obs_names)}-dim obs reconstructed from sequences")

    # Patch the global OBS_NAMES used by all analysis functions
    global OBS_NAMES
    OBS_NAMES = obs_names

    n = len(obs)

    print("\n[obs] Running 7 analyses...")

    corr_rw  = analysis_obs_reward_correlation(obs, rewards, out_dir, args.show)
    print("[obs] 1/7 complete — obs–reward correlation")

    corr_act = analysis_obs_action_correlation(obs, actions, out_dir, args.show)
    print("[obs] 2/7 complete — obs–action correlation")

    delta_r2, r2_1, r2_2 = analysis_markov_test(obs, seqids, out_dir, args.show)
    print("[obs] 3/7 complete — Markov test")

    n90, n95, n99, var_ratio = analysis_pca_redundancy(obs, out_dir, args.show)
    print("[obs] 4/7 complete — PCA redundancy")

    f_reach, f_succ = analysis_label_discriminability(obs, mask, labels, out_dir, args.show)
    print("[obs] 5/7 complete — label discriminability")

    r2_per_label, r2_global = analysis_reward_predictability(obs, total, labels, out_dir, args.show)
    print("[obs] 6/7 complete — reward predictability")

    ac = analysis_temporal_autocorrelation(obs, seqids, out_dir, args.show)
    print("[obs] 7/7 complete — temporal autocorrelation")

    write_report(
        out_dir, n,
        corr_rw, corr_act,
        delta_r2, r2_1, r2_2,
        n90, n95, n99,
        f_reach, f_succ,
        r2_per_label, r2_global,
        ac,
    )

    print(f"\n[obs] Done. Output directory: {out_dir}/")


if __name__ == "__main__":
    main()
