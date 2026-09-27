#!/usr/bin/env python3
"""
G1 MuJoCo Motion Replay — 纯运动学回放（无物理）。
先验证关节映射和轨迹正确，再考虑加物理。

Example:
    python scripts/replay_motions.py --motion-dir PATH --list
    python scripts/replay_motions.py --model-xml PATH --motion-dir PATH
"""

import os
import sys
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np


CSV_HZ = 50
SIM_DT = 0.005
STEPS_PER_FRAME = int(1.0 / CSV_HZ / SIM_DT)  # 4

# g1_29dof_rev_1_0.xml: joints 1-29 = all 29 body DOFs (no hands)
# CSV columns 0..28 map directly to joints 1..29
BODY_JOINT_IDS = list(range(1, 30))  # 29 body joints

# IsaacLab ↔ MuJoCo mapping
MJ_TO_IL = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
     2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int32,
)
IL_TO_MJ = np.argsort(MJ_TO_IL)

# Default joint angles from C++ policy_parameters.hpp (MuJoCo order, radians)
# target = action * action_scale + default_angle
# These are the crouched standing pose offsets applied by the deployment pipeline.
DEFAULT_ANGLES_MJ = np.array([
    -0.312,  # left_hip_pitch_joint
     0.0,    # left_hip_roll_joint
     0.0,    # left_hip_yaw_joint
     0.669,  # left_knee_joint
    -0.363,  # left_ankle_pitch_joint
     0.0,    # left_ankle_roll_joint
    -0.312,  # right_hip_pitch_joint
     0.0,    # right_hip_roll_joint
     0.0,    # right_hip_yaw_joint
     0.669,  # right_knee_joint
    -0.363,  # right_ankle_pitch_joint
     0.0,    # right_ankle_roll_joint
     0.0,    # waist_yaw_joint
     0.0,    # waist_roll_joint
     0.0,    # waist_pitch_joint
     0.2,    # left_shoulder_pitch_joint
     0.2,    # left_shoulder_roll_joint
     0.0,    # left_shoulder_yaw_joint
     0.6,    # left_elbow_joint
     0.0,    # left_wrist_roll_joint
     0.0,    # left_wrist_pitch_joint
     0.0,    # left_wrist_yaw_joint
     0.2,    # right_shoulder_pitch_joint
    -0.2,    # right_shoulder_roll_joint
     0.0,    # right_shoulder_yaw_joint
     0.6,    # right_elbow_joint
     0.0,    # right_wrist_roll_joint
     0.0,    # right_wrist_pitch_joint
     0.0,    # right_wrist_yaw_joint
], dtype=np.float32)


def load_motions(dirs):
    motions = OrderedDict()
    for base in dirs:
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            subdir = os.path.join(base, entry)
            if not os.path.isdir(subdir):
                continue
            csv_path = os.path.join(subdir, "joint_pos.csv")
            if not os.path.isfile(csv_path):
                continue
            try:
                data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                if data.shape[1] != 29:
                    print(f"  [SKIP] {entry}: {data.shape[1]} cols != 29")
                    continue
                motions[entry] = (csv_path, data, data.shape[0])
            except Exception as e:
                print(f"  [SKIP] {entry}: {e}")
    return motions


def load_optional_csv(csv_path, suffix):
    fpath = os.path.join(os.path.dirname(csv_path), suffix)
    if os.path.isfile(fpath):
        return np.loadtxt(fpath, delimiter=",", skiprows=1, dtype=np.float32)
    return None


def reorder_csv(csv_data, csv_order):
    """Reorder CSV columns from csv_order → MuJoCo body joint order."""
    if csv_order == "mj":
        return csv_data  # already in MJ order
    else:
        # csv_data columns 0..28 are IL order → reorder to MJ order
        # IL[i] = MJ[MJ_TO_IL[i]], so MJ[j] = IL[index where MJ_TO_IL[idx]=j]
        # i.e., MJ[j] = csv[:, MJ_TO_IL[j]]
        return csv_data[:, MJ_TO_IL]


def main():
    import argparse
    p = argparse.ArgumentParser(description="G1 MuJoCo Motion Replay (Kinematic)")
    p.add_argument("--motion", type=str, default=None)
    p.add_argument("--motion-dir", type=Path, nargs="+", required=True)
    p.add_argument("--model-xml", type=Path, help="G1 MuJoCo model XML (required to replay)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--loop", action="store_true", default=True)
    p.add_argument("--no-loop", action="store_false", dest="loop")
    p.add_argument("--csv-order", type=str, default="il", choices=["mj", "il"],
                   help="CSV joint order: 'il'=IsaacLab order (default, matches deploy CSVs), 'mj'=MuJoCo body order")
    p.add_argument("--add-default-angles", action="store_true",
                   help="Add C++ default_angles bias to CSV values (for deploy CSVs that lack default pose)")
    args = p.parse_args()

    print("Scanning motions...")
    motions = load_motions(args.motion_dir)
    if not motions:
        print("ERROR: No motions found!")
        sys.exit(1)
    names = list(motions.keys())

    if args.list:
        print(f"\n{len(motions)} motions:")
        for i, n in enumerate(names):
            _, d, nf = motions[n]
            print(f"  {i+1:3d}. {n:45s} {nf:5d}f  ({nf/50:.1f}s)")
        return

    if args.model_xml is None:
        p.error("--model-xml is required to replay")
    import mujoco
    import mujoco.viewer
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    data = mujoco.MjData(model)
    body_qpos_adrs = [model.jnt_qposadr[j] for j in BODY_JOINT_IDS]

    # Select motion
    if args.motion:
        if args.motion in motions:
            idx = names.index(args.motion)
        else:
            matches = [n for n in names if args.motion.lower() in n.lower()]
            if len(matches) == 1:
                idx = names.index(matches[0])
            elif len(matches) > 1:
                print(f"Multiple matches: {matches}")
                return
            else:
                print(f"'{args.motion}' not found. Available: {names}")
                return
    else:
        idx = 0

    csv_path, csv_raw, n_frames = motions[names[idx]]
    csv_data = reorder_csv(csv_raw, args.csv_order)
    body_pos = load_optional_csv(csv_path, "body_pos.csv")
    body_quat = load_optional_csv(csv_path, "body_quat.csv")
    has_root = body_pos is not None and body_quat is not None
    frame = 0
    paused = False

    print(f"\nPlaying [{idx+1}/{len(names)}] {names[idx]}")
    print(f"  Frames: {n_frames}  Duration: {n_frames/50:.1f}s")
    print(f"  CSV order: {args.csv_order}  Root data: {'yes' if has_root else 'no'}")
    print("Keys: Backspace=restart  Up=prev  Down=next  Space=pause\n")

    # Key state
    ks = {"space": False, "up": False, "down": False, "backspace": False}
    def key_cb(keycode):
        import glfw
        for k, code in [("backspace", glfw.KEY_BACKSPACE), ("up", glfw.KEY_UP),
                         ("down", glfw.KEY_DOWN), ("space", glfw.KEY_SPACE)]:
            if keycode == code:
                ks[k] = True

    def switch_motion(new_idx):
        nonlocal idx, csv_path, csv_raw, csv_data, n_frames, frame, body_pos, body_quat, has_root
        idx = new_idx
        csv_path, csv_raw, n_frames = motions[names[idx]]
        csv_data = reorder_csv(csv_raw, args.csv_order)
        body_pos = load_optional_csv(csv_path, "body_pos.csv")
        body_quat = load_optional_csv(csv_path, "body_quat.csv")
        has_root = body_pos is not None and body_quat is not None
        frame = 0
        print(f"  [{idx+1}/{len(names)}] {names[idx]} ({n_frames}f)  root={'yes' if has_root else 'no'}")

    sim_step = 0
    t_start = time.perf_counter()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -15
        viewer.cam.distance = 3.0

        while viewer.is_running():
            # Keys
            if ks["backspace"]:
                ks["backspace"] = False; frame = 0
                print(f"  RESTART [{idx+1}] {names[idx]}")
            if ks["up"]:
                ks["up"] = False; switch_motion((idx - 1) % len(names))
            if ks["down"]:
                ks["down"] = False; switch_motion((idx + 1) % len(names))
            if ks["space"]:
                ks["space"] = False; paused = not paused
                print(f"  {'PAUSED' if paused else 'PLAYING'}")

            if paused:
                viewer.sync()
                time.sleep(0.01)
                continue

            step_in_frame = sim_step % STEPS_PER_FRAME

            # === Set full pose from CSV (pure kinematics) ===
            csv_row = csv_data[frame]

            # Floating base
            if has_root:
                data.qpos[0:3] = body_pos[frame, :3]
                data.qpos[3:7] = body_quat[frame, :4]
            else:
                # Keep root at default standing height
                data.qpos[2] = 0.82
                data.qpos[3] = 1.0

            # Body joints (29 DOF, 1:1 with CSV columns)
            for i, adr in enumerate(body_qpos_adrs):
                val = float(csv_row[i])
                if args.add_default_angles:
                    val += DEFAULT_ANGLES_MJ[i]
                data.qpos[adr] = val

            # Zero all velocities
            data.qvel[:] = 0.0

            # Forward kinematics (no physics)
            mujoco.mj_forward(model, data)

            # Advance frame
            sim_step += 1
            if step_in_frame == STEPS_PER_FRAME - 1:
                frame += 1
                if frame >= n_frames:
                    if args.loop:
                        frame = 0
                    else:
                        frame = n_frames - 1
                        paused = True
                        print("  END (paused)")

            viewer.sync()
            elapsed = time.perf_counter() - t_start
            target = sim_step * SIM_DT
            if target > elapsed:
                time.sleep(target - elapsed)

    print("Done.")


if __name__ == "__main__":
    main()
