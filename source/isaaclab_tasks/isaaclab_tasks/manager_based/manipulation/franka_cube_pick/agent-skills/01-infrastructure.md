---
name: isaaclab-infrastructure
description: Set up the OS environment and Python package manager for Isaac Lab development on a Horde DGXC VM (Ubuntu 22.04, NVIDIA GPU).
level: 1
status: approved
depends_on: []
extends: null
---

## Preconditions

- Ubuntu 22.04 x86_64
- NVIDIA GPU with driver >= 580.65.06
- Internet access to public registries (PyPI, conda-forge)
- User: `horde`, home: `/home/horde`

## Context

Isaac Sim 5.1.0 requires Python 3.11 exactly. The system Python is 3.10 on Ubuntu 22.04,
so an isolated environment is mandatory. We use micromamba instead of conda because it is
faster to install (single binary, no base env overhead) and fully conda-compatible.

cmake and build-essential are required by `isaaclab.sh --install` to compile some extensions.

## Steps

1. **Install micromamba**
   - Command: `curl -L micro.mamba.pm/install.sh | bash -s -- -p ~/micromamba -s bash -n yes`
   - Expected: binary at `~/.local/bin/micromamba`, init block added to `~/.bashrc`

2. **Activate micromamba in the current shell**
   - Command: `source ~/.bashrc`
   - Or for one-off use: `eval "$(~/.local/bin/micromamba shell hook --shell bash)"`

3. **Create the Python 3.11 environment**
   - Command: `micromamba create -n env_isaaclab python=3.11 -c conda-forge -y`
   - Expected: env at `~/micromamba/envs/env_isaaclab/`

4. **Install OS build dependencies**
   - Command: `sudo apt-get install -y cmake build-essential`
   - Expected: both already at newest version on a fresh VM, exit 0

## Variables

| Variable | Value in this project | What it controls | Safe to change? |
|---|---|---|---|
| ENV_NAME | env_isaaclab | micromamba environment name | Yes — update all subsequent skills |
| PYTHON_VERSION | 3.11 | Required by Isaac Sim 5.1.0 | Only if Isaac Sim version changes |
| MICROMAMBA_PREFIX | ~/micromamba | Root prefix for all envs | Yes — must be consistent |

## Verification

```bash
~/.local/bin/micromamba run -n env_isaaclab python --version
```
Expected output: `Python 3.11.x`

## Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `micromamba: command not found` after `source ~/.bashrc` | Init block not written (non-interactive shell) | Run `eval "$(~/.local/bin/micromamba shell hook --shell bash)"` explicitly |
| `PackagesNotFoundError: python=3.11` | conda-forge channel not specified | Add `-c conda-forge` to the create command |

## Changelog

- 2026-04-08: initial version
