# Technical scope

The competition workflow connected motion preparation, development-machine checks, policy simulation, robot deployment and real-robot debugging on the Unitree G1. The public files cover only the **development-machine** side:

1. `convert_bones_csv_to_deploy.py` converts a Bones-style CSV to four 50 Hz preview CSVs plus metadata. It handles root position/orientation and 29 joint angles/velocities. It does **not** create the full set of files required by the robot deployment stack.
2. `replay_motions.py` previews joint trajectories kinematically in a user-supplied G1 MuJoCo model.
3. `run_policy_sim.py` runs an ONNX encoder/decoder policy with user-supplied model and motion assets in MuJoCo, or supports kinematic comparison without the policy.

The full competition workflow also used model files, motion data, and robot-side software. This repository contains the three development-machine scripts listed above.

As Team Lead / Technical Development for 超能逸仙队, Haojin Lu led the team and contributed to G1 technical work, motion deployment, real-robot testing and debugging. See [provenance and attribution](PROVENANCE.md).
