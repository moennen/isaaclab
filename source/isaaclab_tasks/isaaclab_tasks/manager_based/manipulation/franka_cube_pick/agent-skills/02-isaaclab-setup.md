---
name: isaaclab-setup
description: Install Isaac Sim 5.1.0 and Isaac Lab (all frameworks) inside the micromamba env_isaaclab environment, and fix the warp 1.8.2/1.12.1 conflict for Newton.
level: 2
status: draft
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

4. **Install missing pip packages** (required by the dexsuite branch of Isaac Lab)
   - Command: `~/.local/bin/micromamba run -n env_isaaclab pip install lazy-loader pycollada`
   - Command: `~/.local/bin/micromamba run -n env_isaaclab pip install -e /home/horde/projects/IsaacLab/source/isaaclab_physx`
   - Command: `~/.local/bin/micromamba run -n env_isaaclab pip install -e /home/horde/projects/IsaacLab/source/isaaclab_newton`
   - Note: `isaaclab_newton` pulls in `mujoco-warp 3.5.0.2` and `warp-lang 1.12.1`

5. **Fix warp 1.8.2 / 1.12.1 conflict** (CRITICAL for Newton compatibility)

   **Root cause:** Isaac Sim 5.1.0 ships `omni.warp.core-1.8.2` with a bundled copy of
   warp 1.8.2 inside `extscache/omni.warp.core-1.8.2+lx64/warp/`. Newton (mujoco-warp)
   requires warp >= 1.11.0. When AppLauncher starts, it adds the extension dir to sys.path,
   making `import warp` load warp 1.8.2 instead of pip-installed 1.12.1. The bundled
   `types.py` uses `class array(Array[DType])` — a 1-arg subscript that fails in warp 1.12.1
   (which changed `Array` to require 2 type parameters: dtype and ndim).

   **Fix:** The extension's own `extension.py` is designed to use a symlink to pip-installed
   warp when the bundled directory is absent. Replace the bundled directory with that symlink:

   ```bash
   EXT_DIR="$HOME/micromamba/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.warp.core-1.8.2+lx64"
   mv "$EXT_DIR/warp" "$EXT_DIR/warp_bundled_1.8.2_bak"
   ln -s "../../../warp" "$EXT_DIR/warp"
   ```

   The symlink `../../../warp` resolves to `site-packages/warp` (pip-installed warp 1.12.1).
   The `on_startup` code in `extension.py` will then see the valid symlink and leave it alone.

   **To revert:** `rm "$EXT_DIR/warp" && mv "$EXT_DIR/warp_bundled_1.8.2_bak" "$EXT_DIR/warp"`

## Variables

| Variable | Value in this project | What it controls | Safe to change? |
|---|---|---|---|
| ISAACSIM_VERSION | 5.1.0 | Sim version; dictates Python and torch versions | Cascade update to all version pins |
| ISAACLAB_PATH | /home/horde/projects/IsaacLab | Repo root | Yes — update working dirs |

## Verification

```bash
# Basic imports
~/.local/bin/micromamba run -n env_isaaclab python -c "import isaacsim; print('Isaac Sim OK')"
~/.local/bin/micromamba run -n env_isaaclab python -c "import isaaclab; print('Isaac Lab OK')"

# warp conflict fix
~/.local/bin/micromamba run -n env_isaaclab python -c "import warp as wp; print(f'warp {wp.__version__}')"
# Expected: warp 1.12.1

# AppLauncher with warp 1.12.1 (CRITICAL)
OMNI_KIT_ACCEPT_EULA=yes ~/.local/bin/micromamba run -n env_isaaclab python -c "
from isaaclab.app import AppLauncher
import argparse, sys
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
launcher = AppLauncher(args)
import warp as wp
print(f'WARP_VERSION={wp.__version__}', file=sys.stderr, flush=True)
launcher.app.close()
" 2>&1 | grep "WARP_VERSION"
# Expected: WARP_VERSION=1.12.1
```

## Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: isaacsim` | env not activated or pip installed in wrong env | Re-run with explicit `micromamba run -n env_isaaclab` prefix |
| `ERROR: Could not find a version that satisfies isaacsim==5.1.0` | NVIDIA PyPI index missing | Add `--extra-index-url https://pypi.nvidia.com` |
| `isaaclab.sh --install` fails on cmake step | cmake not installed | Run `sudo apt-get install -y cmake build-essential` |
| `TypeError: Too few arguments for Array; actual 1, expected at least 2` | Warp conflict fix not applied | Run step 5 (symlink fix) |
| AppLauncher hangs | EULA not accepted | Set `OMNI_KIT_ACCEPT_EULA=yes` |
| `ModuleNotFoundError: No module named 'lazy_loader'` | Missing pip dep on dexsuite branch | Run step 4 |

## Changelog

- 2026-04-08: initial version
- 2026-04-08: added step 4 (missing pip deps on dexsuite branch) and step 5 (warp conflict root fix); verification updated; status draft (pending clean-env test)
