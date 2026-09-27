#!/usr/bin/env python3
"""
G1 MuJoCo Simulation with ONNX Policy Inference (CPU).

Runs the encoder (reference motion → token) and decoder (token + state → action)
to control the G1 robot in MuJoCo physics simulation.

Example:
    python scripts/run_policy_sim.py --motion-dir PATH --list
    python scripts/run_policy_sim.py --model-xml PATH --encoder PATH --decoder PATH --motion-dir PATH
"""

import os
import sys
import time
import argparse
from contextlib import nullcontext
from pathlib import Path
from collections import OrderedDict

import numpy as np

CSV_HZ = 50
ENC_STEP = 5  # encoder runs at 10Hz (every 5 CSV frames at 50Hz)
# STEPS_PER_FRAME and SIM_DT are computed from the model's actual timestep

# Reorder indices from the repository's G1 mapping definitions:
#   q_mj = q_il[IL_TO_MJ]
#   q_il = q_mj[MJ_TO_IL]
IL_TO_MJ = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
     2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int32,
)
MJ_TO_IL = np.argsort(IL_TO_MJ)

# Per-joint PID gains (MJ order), computed from C++ policy_parameters.hpp
# stiffness = armature * omega^2,  damping = 2 * zeta * armature * omega
# omega = 10*2*pi, zeta = 2.0
_NATURAL_FREQ = 10.0 * 2.0 * np.pi
_A5020 = 0.003609725; _A7520_14 = 0.010177520; _A7520_22 = 0.025101925; _A4010 = 0.00425
_S = {n: a * _NATURAL_FREQ**2 for n, a in [('5020',_A5020),('7520_14',_A7520_14),('7520_22',_A7520_22),('4010',_A4010)]}
_D = {n: 2*2.0*a*_NATURAL_FREQ for n, a in [('5020',_A5020),('7520_14',_A7520_14),('7520_22',_A7520_22),('4010',_A4010)]}
KP_MJ = np.array([_S['7520_22'],_S['7520_22'],_S['7520_14'],_S['7520_22'],2*_S['5020'],2*_S['5020'],
    _S['7520_22'],_S['7520_22'],_S['7520_14'],_S['7520_22'],2*_S['5020'],2*_S['5020'],
    _S['7520_14'],2*_S['5020'],2*_S['5020'],_S['5020'],_S['5020'],_S['5020'],_S['5020'],_S['5020'],_S['4010'],_S['4010'],
    _S['5020'],_S['5020'],_S['5020'],_S['5020'],_S['5020'],_S['4010'],_S['4010']], dtype=np.float32)
KD_MJ = np.array([_D['7520_22'],_D['7520_22'],_D['7520_14'],_D['7520_22'],2*_D['5020'],2*_D['5020'],
    _D['7520_22'],_D['7520_22'],_D['7520_14'],_D['7520_22'],2*_D['5020'],2*_D['5020'],
    _D['7520_14'],2*_D['5020'],2*_D['5020'],_D['5020'],_D['5020'],_D['5020'],_D['5020'],_D['5020'],_D['4010'],_D['4010'],
    _D['5020'],_D['5020'],_D['5020'],_D['5020'],_D['5020'],_D['4010'],_D['4010']], dtype=np.float32)

# Per-joint action scale (MuJoCo/motor order), from C++ policy_parameters.hpp
# action_scale = 0.25 * effort_limit / (armature * omega^2), omega = 10*2*pi
ACTION_SCALE_MJ = np.array([
    0.3507, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,   # left leg
    0.3507, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,   # right leg
    0.5475, 0.4386, 0.4386,                            # waist
    0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,  # left arm
    0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,  # right arm
], dtype=np.float32)

# Default joint angles (crouched standing pose, from policy_parameters.hpp)
DEFAULT_ANGLES_MJ = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,     # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,     # right leg
    0.0, 0.0, 0.0,                              # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,        # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,       # right arm
], dtype=np.float32)


def quat_to_rot6(qw, qx, qy, qz):
    """Convert wxyz quaternion to the first two rotation-matrix columns."""
    return np.array([
        1 - 2*qy*qy - 2*qz*qz,  2*qx*qy - 2*qz*qw,
        2*qx*qy + 2*qz*qw,      1 - 2*qx*qx - 2*qz*qz,
        2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,
    ], dtype=np.float32)


def projected_gravity(qw, qx, qy, qz):
    """World gravity direction expressed in the floating-base frame."""
    return np.array([
        -2.0 * (qx * qz - qy * qw),
        -2.0 * (qy * qz + qx * qw),
        -(1.0 - 2.0 * qx * qx - 2.0 * qy * qy),
    ], dtype=np.float32)


def load_motions(dirs):
    """Scan for motion directories containing joint_pos.csv."""
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
                    continue
                motions[entry] = (csv_path, data, data.shape[0])
            except Exception:
                pass
    return motions


def load_optional_csv(csv_path, suffix):
    """Load an optional CSV from the same directory."""
    fpath = os.path.join(os.path.dirname(csv_path), suffix)
    if os.path.isfile(fpath):
        return np.loadtxt(fpath, delimiter=",", skiprows=1, dtype=np.float32)
    return None


class PolicyInference:
    """Runs the ONNX encoder and decoder for G1 policy inference."""

    def __init__(self, encoder_path, decoder_path):
        import onnxruntime as ort
        print("Loading ONNX models (CPU)...")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        self.encoder = ort.InferenceSession(str(encoder_path), sess_opts)
        self.decoder = ort.InferenceSession(str(decoder_path), sess_opts)

        self.enc_input_dim = self.encoder.get_inputs()[0].shape[1]  # 1751
        self.dec_input_dim = self.decoder.get_inputs()[0].shape[1]  # 994
        self.token_dim = 64
        print(f"  Encoder input: {self.enc_input_dim} dims")
        print(f"  Decoder input: {self.dec_input_dim} dims")
        print(f"  Token dim:     {self.token_dim}")

        # Decoder proprioception history buffers (10 frames at 50Hz step=1)
        # Layout: [base_ang_vel(30) | joint_pos(290) | joint_vel(290) |
        #          action(290) | gravity(30)]
        self.history_size = 10
        self.joint_pos_history = np.zeros((self.history_size, 29), dtype=np.float32)
        self.joint_vel_history = np.zeros((self.history_size, 29), dtype=np.float32)
        self.action_history = np.zeros((self.history_size, 29), dtype=np.float32)
        self.base_ang_vel_history = np.zeros((self.history_size, 3), dtype=np.float32)
        self.gravity_history = np.zeros((self.history_size, 3), dtype=np.float32)

    def precompute_tokens(self, motion_il, body_quat, body_pos=None, joint_vel_il=None):
        """
        Pre-compute encoder tokens for each encoder frame of the reference motion.

        Encoder runs at 10Hz (every ENC_STEP=5 CSV frames at 50Hz).
        Token for frame t is computed from reference data at frames:
        t, t+5, t+10, ..., t+45 (10 frames at step=5).

        NOTE: Encoder expects joint data in IL (IsaacLab) order, matching training.

        Encoder input layout (empirically verified):
          pos 0:    encoder_index = 0.0 (g1 mode)
          pos 1:    encoder_index tokenizer obs = 0.0
          pos 2-291: command_multi_future_nonflat (10 frames × 29 pos, IL order)
          pos 292-581: command_multi_future_nonflat cont'd (10 frames × 29 vel)
          pos 582-591: command_z_multi_future_nonflat (10 frames × 1 root_z)
          pos 592-651: motion_anchor_ori_b_mf_nonflat (10 frames × 6D rotmat)
          pos 652-1750: zeros (teleop + smpl observations, unused by g1)

        Returns:
            tokens: (n_enc_frames, 64) array of precomputed tokens
        """
        n_csv_frames = motion_il.shape[0]
        n_enc_frames = max(1, (n_csv_frames + ENC_STEP - 1) // ENC_STEP)

        tokens = np.zeros((n_enc_frames, self.token_dim), dtype=np.float32)

        # Prefer the motion package's velocity data. Fall back to a central
        # difference for older packages that only contain joint positions.
        if joint_vel_il is None or joint_vel_il.shape != motion_il.shape:
            joint_vel_il = np.empty_like(motion_il)
            joint_vel_il[0] = (motion_il[1] - motion_il[0]) * CSV_HZ
            joint_vel_il[-1] = (motion_il[-1] - motion_il[-2]) * CSV_HZ
            joint_vel_il[1:-1] = (motion_il[2:] - motion_il[:-2]) * (0.5 * CSV_HZ)

        # Root z positions
        root_z = body_pos[:, 2].copy() if body_pos is not None else None

        # Precompute pelvis rot6 from body_quat for all frames
        pelvis_rot6 = None
        if body_quat is not None:
            pelvis_rot6 = np.zeros((body_quat.shape[0], 6), dtype=np.float32)
            for i in range(body_quat.shape[0]):
                q = body_quat[i, :4]  # w,x,y,z
                pelvis_rot6[i] = quat_to_rot6(q[0], q[1], q[2], q[3])

        print(f"  Precomputing tokens: {n_enc_frames} encoder frames "
              f"({n_csv_frames} CSV frames, step={ENC_STEP})")

        t_start = time.perf_counter()
        for enc_t in range(n_enc_frames):
            csv_t = enc_t * ENC_STEP  # corresponding CSV frame
            enc_input = self._build_encoder_input(
                csv_t, motion_il, joint_vel_il, root_z, pelvis_rot6
            )
            outputs = self.encoder.run(None, {"obs_dict": enc_input})
            tokens[enc_t] = outputs[0][0]

        elapsed = time.perf_counter() - t_start
        print(f"  Done in {elapsed:.2f}s ({n_enc_frames/elapsed:.0f} tok/s)")
        return tokens

    def _build_encoder_input(self, t, joint_pos_il, joint_vel_il, root_z, pelvis_rot6):
        """
        Build 1751-dim encoder input for reference frame t (IL order).

        Uses 10 future frames at step=5 (10Hz sampling).
        Frames: t, t+5, t+10, ..., t+45

        Verified layout:
          pos 0:    encoder_index = 0.0 (g1 mode)
          pos 1:    encoder_index tokenizer obs = 0.0
          pos 2-291:  pos_f0..pos_f9  (10 frames × 29, IL order)
          pos 292-581: vel_f0..vel_f9  (10 frames × 29, IL order)
          pos 582-591: root_z_f0..root_z_f9  (10 frames × 1)
          pos 592-651: rot6_f0..rot6_f9  (10 frames × 6)
          pos 652-1750: zeros (teleop + smpl observations)
        """
        enc_input = np.zeros((1, self.enc_input_dim), dtype=np.float32)
        max_frame = joint_pos_il.shape[0] - 1

        # Position 0: encoder_index = 0 (g1 mode)
        enc_input[0, 0] = 0.0

        # Position 1: encoder_index in tokenizer obs
        enc_input[0, 1] = 0.0

        # Positions 2-291: command_multi_future_nonflat — position frames
        for f in range(10):
            fi = min(t + f * ENC_STEP, max_frame)
            off = 2 + f * 29
            enc_input[0, off:off+29] = joint_pos_il[fi]

        # Positions 292-581: command_multi_future_nonflat — velocity frames
        for f in range(10):
            fi = min(t + f * ENC_STEP, max_frame)
            off = 2 + 290 + f * 29
            enc_input[0, off:off+29] = joint_vel_il[fi]

        # Positions 582-591: command_z_multi_future_nonflat — root z
        if root_z is not None:
            max_z = root_z.shape[0] - 1
            for f in range(10):
                fi = min(t + f * ENC_STEP, max_z)
                enc_input[0, 582 + f] = root_z[fi]

        # Positions 592-651: motion_anchor_ori_b_mf_nonflat — rot6
        if pelvis_rot6 is not None:
            max_q = pelvis_rot6.shape[0] - 1
            for f in range(10):
                fi = min(t + f * ENC_STEP, max_q)
                off = 592 + f * 6
                enc_input[0, off:off+6] = pelvis_rot6[fi]

        return enc_input

    def decode_action(self, token, joint_pos_mj, joint_vel_mj, base_ang_vel, gravity_dir):
        """
        Run the decoder to get a 29-dim action.

        Decoder input layout:
          [token(64) | proprioception(930)]
        proprioception:
          [base_ang_vel_hist(30) | joint_pos_hist(290) | joint_vel_hist(290) |
           action_hist(290) | gravity_hist(30)]

        NOTE: proprioception history uses IL order (matching training).
        The decoder outputs actions in IL order.

        Returns:
            action_mj: (29,) action in MuJoCo body order
        """
        # Deployment logs centered joint positions in IsaacLab order.
        joint_pos_il = (joint_pos_mj - DEFAULT_ANGLES_MJ)[MJ_TO_IL]
        joint_vel_il = joint_vel_mj[MJ_TO_IL]

        # Roll history buffers (IL order)
        self.joint_pos_history[:-1] = self.joint_pos_history[1:]
        self.joint_pos_history[-1] = joint_pos_il
        self.joint_vel_history[:-1] = self.joint_vel_history[1:]
        self.joint_vel_history[-1] = joint_vel_il
        self.base_ang_vel_history[:-1] = self.base_ang_vel_history[1:]
        self.base_ang_vel_history[-1] = base_ang_vel
        self.gravity_history[:-1] = self.gravity_history[1:]
        self.gravity_history[-1] = gravity_dir

        # Build proprioception: flatten histories (930 dims)
        proprio = np.concatenate([
            self.base_ang_vel_history.flatten(),    # 30
            self.joint_pos_history.flatten(),       # 290
            self.joint_vel_history.flatten(),       # 290
            self.action_history.flatten(),          # 290
            self.gravity_history.flatten(),         # 30
        ]).astype(np.float32)

        dec_input = np.concatenate(
            [token, proprio]
        ).reshape(1, -1).astype(np.float32)

        outputs = self.decoder.run(None, {"obs_dict": dec_input})
        action_il = outputs[0][0].astype(np.float32)  # (29,) in IL order

        # Update action history (IL order)
        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1] = action_il

        # C++: target_mj[i] = default_mj[i]
        #                        + action_il[IL_TO_MJ[i]] * scale_mj[i]
        action_mj = action_il[IL_TO_MJ] * ACTION_SCALE_MJ

        return action_mj

    def reset_history(self, joint_pos_mj):
        """Reset history buffers to a consistent initial state."""
        joint_pos_il = (joint_pos_mj - DEFAULT_ANGLES_MJ)[MJ_TO_IL]
        self.joint_pos_history[:] = joint_pos_il
        self.joint_vel_history[:] = 0.0
        self.action_history[:] = 0.0
        self.base_ang_vel_history[:] = 0.0
        self.gravity_history[:] = [0.0, 0.0, -1.0]


def main():
    p = argparse.ArgumentParser(description="G1 Policy-Driven MuJoCo Simulation")
    p.add_argument("--motion", type=str, default=None)
    p.add_argument("--motion-dir", type=Path, nargs="+", required=True)
    p.add_argument("--model-xml", type=Path, help="G1 MuJoCo model XML (required to simulate)")
    p.add_argument("--encoder", type=Path, help="ONNX policy encoder (required with policy)")
    p.add_argument("--decoder", type=Path, help="ONNX policy decoder (required with policy)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--loop", action="store_true", default=True)
    p.add_argument("--no-loop", action="store_false", dest="loop")
    p.add_argument("--no-policy", action="store_true",
                   help="Run without policy (kinematic replay for comparison)")
    p.add_argument("--start-running", action="store_true",
                   help="Start immediately instead of waiting for Space")
    p.add_argument("--headless", action="store_true",
                   help="Run without a viewer (useful for installation checks)")
    p.add_argument("--sim-seconds", type=float, default=None,
                   help="Exit after this many simulated seconds (required with --headless)")
    p.add_argument("--warmup-seconds", type=float, default=2.0,
                   help="Hold frame 0 while the policy stabilizes (default: 2.0)")
    args = p.parse_args()

    if args.headless and (args.sim_seconds is None or args.sim_seconds <= 0):
        p.error("--headless requires --sim-seconds with a value greater than zero")
    if args.sim_seconds is not None and args.sim_seconds <= 0:
        p.error("--sim-seconds must be greater than zero")
    if args.warmup_seconds < 0:
        p.error("--warmup-seconds cannot be negative")

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
        p.error("--model-xml is required to simulate")
    if not args.no_policy and (args.encoder is None or args.decoder is None):
        p.error("--encoder and --decoder are required unless --no-policy is set")
    import mujoco
    import mujoco.viewer
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    data = mujoco.MjData(model)
    body_qpos_adrs = [model.jnt_qposadr[j] for j in range(1, 30)]
    body_dof_adrs = [model.jnt_dofadr[j] for j in range(1, 30)]

    # Select motion
    if args.motion:
        if args.motion in motions:
            idx = names.index(args.motion)
        else:
            matches = [n for n in names if args.motion.lower() in n.lower()]
            if len(matches) == 1:
                idx = names.index(matches[0])
            elif not matches:
                print(f"ERROR: Motion '{args.motion}' was not found. Use --list to see names.")
                sys.exit(2)
            else:
                print(f"ERROR: Motion name is ambiguous; matches: {matches}")
                sys.exit(2)
    else:
        idx = 0

    csv_path, csv_raw, n_frames = motions[names[idx]]
    # csv_raw is IL order (IsaacLab), matching encoder training data
    # motion_mj is MJ order (MuJoCo body joint order)
    motion_il = csv_raw.copy()
    motion_mj = csv_raw[:, IL_TO_MJ].copy()
    body_pos = load_optional_csv(csv_path, "body_pos.csv")
    body_quat = load_optional_csv(csv_path, "body_quat.csv")
    joint_vel_il = load_optional_csv(csv_path, "joint_vel.csv")
    print(f"  Motion: {names[idx]}  Frames: {n_frames}  "
          f"Root: {'yes' if body_pos is not None else 'no'}")

    # Initialize policy inference
    policy = None
    tokens = None
    if not args.no_policy:
        policy = PolicyInference(args.encoder, args.decoder)
        print()
        tokens = policy.precompute_tokens(motion_il, body_quat, body_pos, joint_vel_il)
        n_enc_frames = tokens.shape[0]
        print(f"  Token stats: range=[{tokens.min():.3f}, {tokens.max():.3f}] "
              f"mean={tokens.mean():+.4f}")

    print(f"\nSimulating [{idx+1}/{len(names)}] {names[idx]}")
    print(f"  Duration: {n_frames/50:.1f}s")
    if policy:
        print(f"  Mode: ONNX policy inference (CPU)")
    else:
        print(f"  Mode: Kinematic replay (no policy)")
    if not args.headless:
        print("Keys: Space=start/pause  Backspace=restart  Up=prev  Down=next\n")

    frame = 0
    # Opening in a known, stationary state makes inspecting a modified motion safer.
    # Headless checks always run immediately because there is no keyboard input.
    paused = not args.start_running and not args.headless
    ks = {"space": False, "up": False, "down": False, "backspace": False}

    def key_cb(keycode):
        import glfw
        for k, code in [("backspace", glfw.KEY_BACKSPACE),
                         ("up", glfw.KEY_UP),
                         ("down", glfw.KEY_DOWN),
                         ("space", glfw.KEY_SPACE)]:
            if keycode == code:
                ks[k] = True

    def switch_motion(new_idx):
        nonlocal idx, csv_path, csv_raw, motion_il, motion_mj, n_frames, frame
        nonlocal body_pos, body_quat, joint_vel_il, tokens, n_enc_frames, policy
        idx = new_idx
        csv_path, csv_raw, n_frames = motions[names[idx]]
        motion_il = csv_raw.copy()
        motion_mj = csv_raw[:, IL_TO_MJ].copy()
        body_pos = load_optional_csv(csv_path, "body_pos.csv")
        body_quat = load_optional_csv(csv_path, "body_quat.csv")
        joint_vel_il = load_optional_csv(csv_path, "joint_vel.csv")
        frame = 0
        if policy:
            tokens = policy.precompute_tokens(motion_il, body_quat, body_pos, joint_vel_il)
            n_enc_frames = tokens.shape[0]
        print(f"  [{idx+1}/{len(names)}] {names[idx]} ({n_frames}f)")

    sim_step = 0

    # Compute steps per frame from the model's actual timestep
    # The XML model has timestep=0.002 (500 Hz); CSV is at 50 Hz → 10 sim steps/frame
    global STEPS_PER_FRAME
    STEPS_PER_FRAME = int(1.0 / CSV_HZ / model.opt.timestep)
    print(f"  Model timestep: {model.opt.timestep*1000:.1f}ms, "
          f"steps/frame: {STEPS_PER_FRAME}")

    # Initialize robot to crouched standing pose
    # Set root height so feet touch the ground (computed from geometric model)
    data.qpos[2] = 0.82   # standing height (approximate, will settle on ground)
    data.qpos[3] = 1.0    # quat w
    for i, adr in enumerate(body_qpos_adrs):
        data.qpos[adr] = float(DEFAULT_ANGLES_MJ[i])
    mujoco.mj_forward(model, data)

    # Reset policy history to initial robot state
    if policy:
        init_qpos = np.array([data.qpos[adr] for adr in body_qpos_adrs],
                             dtype=np.float32)
        policy.reset_history(init_qpos)

    # The decoder is a 50 Hz controller. MuJoCo runs at 500 Hz, so its target
    # must be held for STEPS_PER_FRAME physics steps.
    target_qpos = DEFAULT_ANGLES_MJ.copy()
    warmup_ticks_remaining = (
        int(np.ceil(args.warmup_seconds * CSV_HZ)) if policy is not None else 0
    )

    max_sim_steps = None
    if args.sim_seconds is not None:
        max_sim_steps = int(np.ceil(args.sim_seconds / model.opt.timestep))

    viewer_context = (
        nullcontext(None)
        if args.headless
        else mujoco.viewer.launch_passive(model, data, key_callback=key_cb)
    )

    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -15
            viewer.cam.distance = 3.0
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body("pelvis").id
            if paused:
                print("  READY (paused) - press Space to start")
            elif warmup_ticks_remaining:
                print(f"  STABILIZING at frame 0 ({warmup_ticks_remaining / CSV_HZ:.1f}s)")
            else:
                print("  PLAYING")

        next_step_deadline = time.perf_counter()

        while ((viewer is None or viewer.is_running()) and
               (max_sim_steps is None or sim_step < max_sim_steps)):
            # --- Key handling ---
            if ks["backspace"]:
                ks["backspace"] = False
                frame = 0
                if policy:
                    init_qpos = np.array([data.qpos[adr] for adr in body_qpos_adrs],
                                         dtype=np.float32)
                    policy.reset_history(init_qpos)
                print(f"  RESTART [{idx+1}] {names[idx]}")
            if ks["up"]:
                ks["up"] = False
                switch_motion((idx - 1) % len(names))
            if ks["down"]:
                ks["down"] = False
                switch_motion((idx + 1) % len(names))
            if ks["space"]:
                ks["space"] = False
                paused = not paused
                if paused:
                    print("  PAUSED")
                elif warmup_ticks_remaining:
                    print(f"  STABILIZING at frame 0 ({warmup_ticks_remaining / CSV_HZ:.1f}s)")
                else:
                    print("  PLAYING")

            if paused:
                if viewer is None:
                    break
                viewer.sync()
                time.sleep(0.01)
                # Do not try to catch up all paused wall-clock time on resume.
                next_step_deadline = time.perf_counter()
                continue

            # --- Reference pose for current frame ---
            csv_row = motion_mj[frame]  # MJ order

            # --- Set floating base ---
            if not args.no_policy:
                # Policy mode: let physics control the floating base.
                # Only set initial root pose (done once at startup), don't override.
                # Gravity and ground contact handle the rest.
                pass
            elif body_pos is not None and body_quat is not None:
                # Kinematic mode: directly set root from reference trajectory
                data.qpos[0:3] = body_pos[frame, :3]
                data.qpos[3:7] = body_quat[frame, :4]
            else:
                # Kinematic mode, no root data: hold at standing height
                data.qpos[2] = 0.82
                data.qpos[3] = 1.0

            if args.no_policy:
                # Pure kinematic comparison: show the reference exactly, without
                # mixing a prescribed floating base with physics integration.
                for i, adr in enumerate(body_qpos_adrs):
                    data.qpos[adr] = float(csv_row[i])
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
            else:
                # --- Get current joint state ---
                current_qpos_mj = np.array(
                    [data.qpos[adr] for adr in body_qpos_adrs], dtype=np.float32)
                current_qvel_mj = np.array(
                    [data.qvel[adr] for adr in body_dof_adrs], dtype=np.float32)

                # --- Base angular velocity (in body frame, approximate) ---
                # Use gyro from sensor data if available, else approximate.
                base_ang_vel = np.zeros(3, dtype=np.float32)
                if model.nsensor > 0:
                    for s in range(model.nsensor):
                        if model.sensor_type[s] == mujoco.mjtSensor.mjSENS_GYRO:
                            adr = model.sensor_adr[s]
                            base_ang_vel = data.sensordata[adr:adr+3].copy()
                            break
                if np.all(base_ang_vel == 0):
                    base_ang_vel = data.qvel[3:6].copy()
                base_quat = data.qpos[3:7]
                gravity_dir = projected_gravity(
                    base_quat[0], base_quat[1], base_quat[2], base_quat[3]
                )

                if sim_step % STEPS_PER_FRAME == 0:
                    # Run the decoder once per 50 Hz control tick and hold its
                    # target throughout the ten 500 Hz physics substeps.
                    enc_frame = frame // ENC_STEP
                    enc_frame = min(enc_frame, tokens.shape[0] - 1)
                    token = tokens[enc_frame]
                    action_mj = policy.decode_action(
                        token, current_qpos_mj, current_qvel_mj, base_ang_vel, gravity_dir
                    )
                    target_qpos = action_mj + DEFAULT_ANGLES_MJ

                # --- PD control with gravity compensation ---
                tau = (KP_MJ * (target_qpos - current_qpos_mj) +
                       KD_MJ * (0.0 - current_qvel_mj))
                mujoco.mj_rne(model, data, 0, data.qfrc_bias)

                # MuJoCo motor actuators consume torque commands.
                for i in range(29):
                    data.ctrl[i] = float(tau[i]) + data.qfrc_bias[body_dof_adrs[i]]
                mujoco.mj_step(model, data)

            # --- Advance frame ---
            sim_step += 1
            step_in_frame = sim_step % STEPS_PER_FRAME
            if step_in_frame == 0:
                if warmup_ticks_remaining > 0:
                    warmup_ticks_remaining -= 1
                    if warmup_ticks_remaining == 0:
                        print("  STABILIZED - PLAYING reference motion")
                else:
                    frame += 1
                if frame >= n_frames:
                    if args.loop:
                        frame = 0
                        if policy:
                            init_qpos = np.array(
                                [data.qpos[adr] for adr in body_qpos_adrs],
                                dtype=np.float32)
                            policy.reset_history(init_qpos)
                    else:
                        frame = n_frames - 1
                        paused = True
                        print("  END (paused)")

            # --- Sync with real time ---
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError(f"Simulation became non-finite at step {sim_step}")

            if viewer is not None:
                viewer.sync()
                next_step_deadline += model.opt.timestep
                sleep_for = next_step_deadline - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)

    simulated_seconds = sim_step * model.opt.timestep
    final_gravity = projected_gravity(*data.qpos[3:7])
    final_tilt = np.degrees(np.arccos(np.clip(-final_gravity[2], -1.0, 1.0)))
    print(
        f"Done. Simulated {sim_step} steps ({simulated_seconds:.3f}s); state is finite. "
        f"Base z={data.qpos[2]:.3f}m, tilt={final_tilt:.1f}deg."
    )


if __name__ == "__main__":
    main()
