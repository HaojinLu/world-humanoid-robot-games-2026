# Technical scope

The competition workflow connected motion preparation, development-machine checks, policy simulation, robot deployment and real-robot debugging on the Unitree G1. The public files cover only the **development-machine** side:

1. `convert_bones_csv_to_deploy.py` converts a Bones-style CSV to four 50 Hz preview CSVs plus metadata. It handles root position/orientation and 29 joint angles/velocities. It does **not** create the full set of files required by the robot deployment stack.
2. `replay_motions.py` previews joint trajectories kinematically in a user-supplied G1 MuJoCo model.
3. `run_policy_sim.py` runs an ONNX encoder/decoder policy with user-supplied model and motion assets in MuJoCo, or supports kinematic comparison without the policy.

The source archive also contains model files, motion trajectories, third-party code, logs and other scripts. Those are excluded from this release. The separate on-robot source has not been provided for comparison or publication. None of the included scripts alone can reproduce the competition system, and no benchmark, reliability or safety result is claimed.

As Team Lead / Technical Integration for 超能逸仙队, Haojin Lu coordinated technical integration and contributed to G1 motion deployment, real-robot testing and debugging. This role does not imply individual authorship of all team code or the NVIDIA base framework. See [provenance and attribution](PROVENANCE.md).
