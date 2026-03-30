# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_tasks_experimental' python package."""

from setuptools import setup

# Installation operation
setup(
    name="isaaclab_tasks_experimental",
    version="0.0.1",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url="https://github.com/isaac-sim/IsaacLab",
    description="Extension containing suite of experimental environments for robot learning.",
    keywords=["robotics", "rl", "il", "learning"],
    python_requires=">=3.11",
    install_requires=[
        "numpy>2",
        "torch>=2.7",
        "torchvision>=0.14.1",
        "protobuf>=3.20.2,!=5.26.0",
        "tensorboard",
        "scikit-learn",
        "numba",
    ],
    packages=["isaaclab_tasks_experimental"],
    include_package_data=True,
    zip_safe=False,
)
