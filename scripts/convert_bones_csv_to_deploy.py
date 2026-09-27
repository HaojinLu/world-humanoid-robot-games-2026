#!/usr/bin/env python3
"""
convert_bones_csv_to_deploy.py
将 Bones-SEED 原始 CSV (Flat CSV) 转换为运动预览所需的四个 CSV 和元数据。
此脚本不生成完整机器人部署包。
自动处理: 单位转换(cm->m, deg->rad), 关节顺序(MJ->IL), 重采样到50Hz, 计算关节速度。
"""
import os
import csv
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

# IsaacLab ↔ MuJoCo joint reordering (29 DOFs for G1)
MJ_TO_IL = np.array([0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28], dtype=np.int32)
IL_TO_MJ = np.zeros(29, dtype=int)
for mj, il in enumerate(MJ_TO_IL):
    IL_TO_MJ[il] = mj

def process_single_csv(csv_path, out_dir, fps_source):
    name = os.path.splitext(os.path.basename(csv_path))[0]
    motion_dir = os.path.join(out_dir, name)
    if fps_source <= 0:
        raise ValueError("--fps-source must be positive")
    # Detect a numeric first row before pandas can mistake it for a header.
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        first_row = next(csv.reader(handle), [])
    try:
        headerless = len(first_row) == 36 and all(np.isfinite(float(v)) for v in first_row)
    except ValueError:
        headerless = False
    if headerless:
        cols = ['Frame', 'root_translateX', 'root_translateY', 'root_translateZ',
                'root_rotateX', 'root_rotateY', 'root_rotateZ'] + [f"joint_{i}_dof" for i in range(29)]
        df = pd.read_csv(csv_path, header=None, names=cols)
    else:
        df = pd.read_csv(csv_path)
    if len(df) < 2:
        raise ValueError(f"{csv_path}: at least two source frames are required")
    required = ['root_translateX', 'root_translateY', 'root_translateZ',
                'root_rotateX', 'root_rotateY', 'root_rotateZ']
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {missing}")

    # 2. 提取数据
    root_pos_cm = df[['root_translateX', 'root_translateY', 'root_translateZ']].values
    root_pos_m = root_pos_cm / 100.0  # cm -> m

    euler_deg = df[['root_rotateX', 'root_rotateY', 'root_rotateZ']].values

    joint_cols = [c for c in df.columns if c.endswith("_dof")]
    if len(joint_cols) != 29:
        raise ValueError(f"{csv_path}: expected 29 joint columns, found {len(joint_cols)}")
    values = df[required + joint_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{csv_path}: input contains non-finite values")
    joint_pos_deg = df[joint_cols].values
    joint_pos_mj = np.deg2rad(joint_pos_deg) # deg -> rad

    # 3. 重采样到 50Hz (C++ 栈硬性要求)
    fps_target = 50
    N = len(df)
    t_source = np.arange(N) / fps_source
    T_total = t_source[-1]
    if T_total < 1.0 / fps_target:
        raise ValueError(f"{csv_path}: source duration is shorter than one target interval")

    # 目标时间轴 (确保包含最后一帧)
    t_target = np.arange(0, T_total + 1e-6, 1.0 / fps_target)
    t_target = np.clip(t_target, t_source[0], t_source[-1]) # 防止浮点误差越界

    # 插值位置
    interp_pos = interp1d(t_source, root_pos_m, axis=0, kind='linear')
    root_pos_target = interp_pos(t_target)

    # 插值关节角度
    interp_jpos = interp1d(t_source, joint_pos_mj, axis=0, kind='linear')
    joint_pos_mj_target = interp_jpos(t_target)

    # 插值四元数 (Slerp)
    rots_source = Rotation.from_euler('xyz', euler_deg, degrees=True)
    if len(t_source) > 1:
        slerp = Slerp(t_source, rots_source)
        rots_target = slerp(t_target)
    else:
        rots_target = rots_source

    root_quat_xyzw = rots_target.as_quat()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]] # xyzw -> wxyz

    # 4. 关节顺序转换 (MuJoCo -> IsaacLab)
    joint_pos_il = joint_pos_mj_target[:, IL_TO_MJ]

    # 5. 计算关节速度 (rad/s)
    dt = 1.0 / fps_target
    joint_vel_il = np.gradient(joint_pos_il, dt, axis=0)

    # 6. 保存为运动预览 CSV；这不是完整部署数据包。
    os.makedirs(motion_dir, exist_ok=True)
    header_pos = ",".join([f"joint_{i}" for i in range(29)])
    np.savetxt(os.path.join(motion_dir, "joint_pos.csv"), joint_pos_il, delimiter=",", header=header_pos, comments='')

    header_vel = ",".join([f"joint_vel_{i}" for i in range(29)])
    np.savetxt(os.path.join(motion_dir, "joint_vel.csv"), joint_vel_il, delimiter=",", header=header_vel, comments='')

    header_bpos = "body_0_x,body_0_y,body_0_z"
    np.savetxt(os.path.join(motion_dir, "body_pos.csv"), root_pos_target, delimiter=",", header=header_bpos, comments='')

    header_bquat = "body_0_w,body_0_x,body_0_y,body_0_z"
    np.savetxt(os.path.join(motion_dir, "body_quat.csv"), root_quat_wxyz, delimiter=",", header=header_bquat, comments='')

    with open(os.path.join(motion_dir, "metadata.txt"), "w") as f:
        f.write(f"Metadata for: {name}\n")
        f.write("==============================\n")
        f.write("Body part indexes:\n")
        f.write("[0]\n")
        f.write(f"Total timesteps: {len(root_pos_target)}\n")
        f.write(f"FPS: {fps_target}\n")

    print(f"Converted {name}: {N} frames at {fps_source} Hz -> {len(root_pos_target)} frames at 50 Hz: {motion_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory of source Bones CSV files")
    parser.add_argument("--output", required=True, help="Output directory for development preview CSVs")
    parser.add_argument("--fps-source", type=int, default=120, help="Source frame rate (default: 120 Hz)")
    args = parser.parse_args()

    if args.fps_source <= 0:
        parser.error("--fps-source must be positive")
    if not os.path.isdir(args.input):
        parser.error("--input must be an existing directory")
    os.makedirs(args.output, exist_ok=True)
    csv_files = sorted(f for f in os.listdir(args.input) if f.lower().endswith(".csv"))
    print(f"Found {len(csv_files)} CSV files")
    for csv_f in csv_files:
        process_single_csv(os.path.join(args.input, csv_f), args.output, args.fps_source)

    print("Conversion complete. Outputs are for development preview, not robot deployment.")

if __name__ == "__main__":
    main()