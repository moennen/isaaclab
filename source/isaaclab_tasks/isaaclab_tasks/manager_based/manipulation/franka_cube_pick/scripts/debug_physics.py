"""Physics debugger for the Franka cube pick task.

Runs a single scripted episode with per-step diagnostics, focusing on a
configurable time window so the feedback loop is fast (a few seconds of
simulation instead of the full 10 s).

Prints every step: cube position, cube velocity, finger positions, EE position,
and any large sudden displacements (knocks).

Usage
-----
# Default: run 3.0 s → 8.0 s window (covers descent + grasp + early lift)
python debug_physics.py

# Custom window and cube position
python debug_physics.py --t_start 4.0 --t_end 7.5 --cube_x 0.5 --cube_y 0.1

# Print every N steps (reduce output verbosity)
python debug_physics.py --print_every 5

# Run from t=0 (observe full approach)
python debug_physics.py --t_start 0.0 --t_end 5.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as nik

sys.path.insert(0, str(Path(__file__).parent))

# Re-use build_models and solve_ik_single from the generator.
# We import just the pieces we need to avoid circular imports.
from generate_sequences import (
    _N_ROBOT_JOINTS,
    _FINGER_OPEN,
    _FINGER_CLOSE,
    build_models,
    solve_ik_single,
    _finger_centering_shift,
)
from _common.waypoint_ik import WaypointStateMachine, T_GRASP, T_GRIP, T_LIFT
from _common.sampling import sample_cube_pos, sample_label
from _common.sequence_schema import DEFAULT_CONFIG


def debug_episode(
    ik_model,
    phys_model,
    hand_idx: int,
    cube_body_idx: int,
    solver,
    cube_pos: list[float],
    t_start: float,
    t_end: float,
    dt: float,
    print_every: int,
    label: dict,
    cfg: dict,
):
    """Run one episode from t=0, printing diagnostics for t ∈ [t_start, t_end].

    Runs from t=0 so the arm follows the full trajectory (approach → descend →
    grasp → lift), but only prints diagnostics in the requested window to keep
    output manageable.
    """
    import torch
    _ = torch  # used by WaypointStateMachine indirectly

    n_phys = phys_model.joint_coord_count
    n_dof  = phys_model.joint_dof_count

    T_END = max(t_end + 0.5, 10.0)   # always run to at least the end of the window
    steps_per_ep = int(T_END / dt)

    state_0 = phys_model.state()
    state_1 = phys_model.state()
    control = phys_model.control()

    has_collider = hasattr(phys_model, "collider")
    contacts = phys_model.collider() if has_collider else None

    # IK state for FK evaluation
    ik_state_fk = ik_model.state()

    # Reset: robot to default, cube to requested position
    default_robot_q = ik_model.joint_q.numpy()[:_N_ROBOT_JOINTS].astype(np.float32)
    cx, cy, cz = cube_pos

    joint_q_init = np.zeros(n_phys, dtype=np.float32)
    joint_q_init[:_N_ROBOT_JOINTS] = default_robot_q
    joint_q_init[_N_ROBOT_JOINTS:_N_ROBOT_JOINTS + 3] = [cx, cy, cz]
    joint_q_init[_N_ROBOT_JOINTS + 3:n_phys] = [0.0, 0.0, 0.0, 1.0]

    state_0.joint_q.assign(joint_q_init)
    state_0.joint_qd.assign(np.zeros(n_dof, dtype=np.float32))

    cube_init_arr = np.array([cx, cy, cz], dtype=np.float32)

    # Waypoint state machine (mirrors generator)
    import torch as _torch
    cube_pos_t = _torch.tensor([cx, cy, cz])
    device = _torch.device("cpu")
    sm = WaypointStateMachine(cube_pos_b=cube_pos_t, label=label, cfg=cfg, device=device)

    gripper_closed = False
    prev_robot_q  = default_robot_q.copy()
    prev_cube_pos = cube_init_arr.copy()
    cube_pos_arr  = cube_init_arr.copy()
    finger_cmd    = _FINGER_OPEN
    _arm6_locked  = None   # set at T_GRIP to prevent wrist sweep during lift

    print(f"\n{'='*72}")
    print(f"  cube_init=[{cx:.4f},{cy:.4f},{cz:.4f}]  window=[{t_start:.1f}s,{t_end:.1f}s]")
    print(f"  T_GRASP={T_GRASP}s  T_GRIP={T_GRIP}s  T_LIFT={T_LIFT}s")
    print(f"{'='*72}")
    print(f"{'step':>5} {'t':>6} {'cube_x':>8} {'cube_y':>8} {'cube_z':>8} "
          f"{'dcube':>7} {'vx':>7} {'vy':>7} {'vz':>7} "
          f"{'ee_x':>7} {'ee_y':>7} {'ee_z':>7} "
          f"{'f1':>7} {'f2':>7} {'cmd':>5}")
    print('-'*130)

    knock_count = 0

    for step in range(steps_per_ep):
        t = step * dt

        ee_pos_t, quat, finger_cmd = sm.get_target(t)
        ee_target   = ee_pos_t.tolist()
        quat_wxyz   = quat.tolist()

        robot_q_now  = state_0.joint_q.numpy()[:_N_ROBOT_JOINTS].astype(np.float32)
        joint_pos_cmd = solve_ik_single(ik_model, hand_idx, ee_target, robot_q_now, quat_wxyz)

        # Finger centering correction (same as generator)
        if label["reachable"] and label["success"]:
            ee_arr  = np.array(ee_target, dtype=np.float32)
            ee_dist = float(np.linalg.norm(ee_arr - cube_pos_arr))
            if ee_dist < 0.25:
                shift = _finger_centering_shift(
                    ik_model, hand_idx, ik_state_fk, joint_pos_cmd, cube_pos_arr
                )
                if np.linalg.norm(shift) > 0.003:
                    corrected_target = (ee_arr + shift).tolist()
                    joint_pos_cmd = solve_ik_single(
                        ik_model, hand_idx, corrected_target, joint_pos_cmd, None
                    )

        ik_cmd_arm6 = float(joint_pos_cmd[6])   # wrist IK command (before lock)

        # Lock arm[6] at T_GRIP to prevent wrist sweep from knocking cube
        if t >= T_GRIP:
            if _arm6_locked is None:
                _arm6_locked = float(state_0.joint_q.numpy()[6])  # current arm6 before step
            joint_pos_cmd[6] = _arm6_locked

        joint_pos_cmd[7] = finger_cmd
        joint_pos_cmd[8] = finger_cmd

        if finger_cmd < 0.01 and not gripper_closed:
            gripper_closed = True

        cmd_full = np.zeros(n_phys, dtype=np.float32)
        cmd_full[:_N_ROBOT_JOINTS] = joint_pos_cmd
        control.joint_target_pos = wp.array(cmd_full, dtype=wp.float32)

        state_0.clear_forces()
        if contacts is not None:
            solver.step(state_0, state_1, control, contacts, dt)
        else:
            solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

        newton.eval_fk(phys_model, state_0.joint_q, state_0.joint_qd, state_0)
        body_q = state_0.body_q.numpy()
        ee_pos_now   = body_q[hand_idx][:3]
        cube_pos_now = body_q[cube_body_idx][:3]

        all_q  = state_0.joint_q.numpy()
        all_qd = state_0.joint_qd.numpy()

        cube_arr = np.array(cube_pos_now, dtype=np.float32)
        cube_vel = all_qd[_N_ROBOT_JOINTS:_N_ROBOT_JOINTS + 3]
        f1       = float(all_q[7])
        f2       = float(all_q[8])
        arm6     = float(all_q[6])   # wrist rotation joint (panda_joint7)
        cube_delta = float(np.linalg.norm(cube_arr - prev_cube_pos))

        if t_start <= t <= t_end:
            # Flag large displacements
            knock_flag = ""
            if cube_delta > 0.005:
                knock_flag = f"  *** KNOCK +{cube_delta*100:.1f}cm ***"
                knock_count += 1

            should_print = (step % print_every == 0) or (cube_delta > 0.005)
            if should_print:
                print(
                    f"{step:>5} {t:>6.3f}"
                    f" {cube_pos_now[0]:>8.4f} {cube_pos_now[1]:>8.4f} {cube_pos_now[2]:>8.4f}"
                    f" {cube_delta*100:>7.2f}"
                    f" {cube_vel[0]:>7.3f} {cube_vel[1]:>7.3f} {cube_vel[2]:>7.3f}"
                    f" {ee_pos_now[0]:>7.4f} {ee_pos_now[1]:>7.4f} {ee_pos_now[2]:>7.4f}"
                    f" {f1:>7.4f} {f2:>7.4f}"
                    f" a6_act={arm6:>7.4f} a6_cmd={ik_cmd_arm6:>7.4f}"
                    + knock_flag
                )

        prev_cube_pos = cube_arr
        cube_pos_arr  = cube_arr

    peak_cube_z = float(cube_arr[2])
    print('-'*130)
    print(f"\nSummary: knock_count={knock_count}  final_cube_z={peak_cube_z:.4f}m"
          f"  cube_final=[{cube_arr[0]:.4f},{cube_arr[1]:.4f},{cube_arr[2]:.4f}]")
    print(f"  cube_init=[{cx:.4f},{cy:.4f},{cz:.4f}]"
          f"  cube_xy_drift=[{cube_arr[0]-cx:.4f},{cube_arr[1]-cy:.4f}]m")


def main():
    parser = argparse.ArgumentParser(description="Physics debug tool for Franka cube pick")
    parser.add_argument("--cube_x",     type=float, default=0.55,  help="Cube X position [m]")
    parser.add_argument("--cube_y",     type=float, default=0.0,   help="Cube Y position [m]")
    parser.add_argument("--cube_z",     type=float, default=0.025, help="Cube Z position [m] (half-height)")
    parser.add_argument("--t_start",    type=float, default=3.5,   help="Start of print window [s]")
    parser.add_argument("--t_end",      type=float, default=8.0,   help="End of print window [s]")
    parser.add_argument("--dt",         type=float, default=0.02,  help="Physics timestep [s]")
    parser.add_argument("--print_every",     type=int,   default=5,    help="Print every N steps within window")
    parser.add_argument("--grasp_ee_height", type=float, default=0.10, help="EE height above cube centre for grasp [m]")
    args = parser.parse_args()

    from generate_sequences import _find_panda_urdf

    urdf_path = _find_panda_urdf()
    print(f"Loading Panda URDF: {urdf_path}")

    ik_model, phys_model, hand_idx, cube_body_idx, solver = build_models(urdf_path)
    print(f"IK model:   {ik_model.body_count} bodies, {ik_model.joint_coord_count} joint coords")
    print(f"Phys model: {phys_model.body_count} bodies, {phys_model.joint_coord_count} joint coords")
    print(f"Solver:     {type(solver).__name__}")

    cfg = {**DEFAULT_CONFIG, "grasp_ee_height": args.grasp_ee_height}
    label = {"reachable": True, "success": True}   # always test the grasp scenario

    debug_episode(
        ik_model=ik_model,
        phys_model=phys_model,
        hand_idx=hand_idx,
        cube_body_idx=cube_body_idx,
        solver=solver,
        cube_pos=[args.cube_x, args.cube_y, args.cube_z],
        t_start=args.t_start,
        t_end=args.t_end,
        dt=args.dt,
        print_every=args.print_every,
        label=label,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
