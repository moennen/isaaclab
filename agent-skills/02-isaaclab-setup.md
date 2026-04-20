---
name: isaaclab-setup
description: Install Isaac Sim 5.1.0 and Isaac Lab (all frameworks) inside the micromamba env_isaaclab environment.
level: 2
status: approved
depends_on: [isaaclab-infrastructure]
extends: null
---

## Preconditions

- `env_isaaclab` environment exists with Python 3.11 (skill 01 complete)
- Internet access to `pypi.nvidia.com` and `download.pytorch.org`
- GLIBC >= 2.35 (Ubuntu 22.04 ships 2.35 — verify with `ldd --version`)

## Context

Isaac Sim is distributed as pip wheels on the NVIDIA PyPI index. Version 5.1.0
is the current target; it pins torch==2.7.0 and torchvision==0.22.0 automatically
as dependencies, so a separate torch installation is not needed.

The `isaaclab.sh --install` script installs all Isaac Lab extensions and learning
frameworks (rl_games, rsl_rl, sb3, skrl, robomimic). It also re-pins torch to the
CUDA 12.8 variant from the PyTorch index — this is expected and harmless.

Working directory for all steps: `/home/horde/projects/IsaacLab`

## Steps

1. **Upgrade pip**
   - Command: `~/.local/bin/micromamba run -n env_isaaclab pip install --upgrade pip`

2. **Install Isaac Sim**
   - Command: `~/.local/bin/micromamba run -n env_isaaclab pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`
   - Duration: ~5–10 min (large download)
   - Note: minor packaging conflict on `packaging` version is harmless (wheel 0.46.3 vs 23.0)

3. **Install Isaac Lab extensions**
   - Working dir: `/home/horde/projects/IsaacLab`
   - Command: `~/.local/bin/micromamba run -n env_isaaclab bash isaaclab.sh --install`
   - Duration: ~5 min
   - Expected last line: `Successfully installed egl_probe robomimic ...`
   - Note: "Running inside a docker container. Skipping VSCode settings setup." is expected

4. **Install the experimental tasks package (editable)**
   - Working dir: `/home/horde/projects/IsaacLab/source/isaaclab_tasks_experimental`
   - Command: `~/.local/bin/micromamba run -n env_isaaclab pip install -e .`

## Variables

| Variable | Value in this project | What it controls | Safe to change? |
|---|---|---|---|
| ISAACSIM_VERSION | 5.1.0 | Sim version; dictates Python and torch versions | Cascade update to all version pins |
| ISAACLAB_PATH | /home/horde/projects/IsaacLab | Repo root | Yes — update working dirs |

## Verification

```bash
~/.local/bin/micromamba run -n env_isaaclab python -c "import isaacsim; print('Isaac Sim OK')"
~/.local/bin/micromamba run -n env_isaaclab python -c "import isaaclab; print('Isaac Lab OK')"
~/.local/bin/micromamba run -n env_isaaclab python -c "import isaaclab_tasks_experimental; print('Experimental tasks OK')"
```
Each line should print `... OK`.

## Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: isaacsim` | env not activated or pip installed in wrong env | Re-run with explicit `micromamba run -n env_isaaclab` prefix |
| `ERROR: Could not find a version that satisfies isaacsim==5.1.0` | NVIDIA PyPI index missing | Add `--extra-index-url https://pypi.nvidia.com` |
| `isaaclab.sh --install` fails on cmake step | cmake not installed | Run `sudo apt-get install -y cmake build-essential` |

## Changelog

- 2026-04-08: initial version
