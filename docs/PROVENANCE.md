# Code provenance and attribution

These three development-machine scripts come from 超能逸仙队’s 2026 competition handover. The handover does not identify individual authors for the files, so they are attributed to the **team**.

| Public file | Source in the handover | Public-release changes |
| --- | --- | --- |
| `scripts/run_policy_sim.py` | `run_policy_sim.py` | Removed collaborator-specific paths and made model, ONNX and motion paths explicit; `--list` runs without model assets. |
| `scripts/replay_motions.py` | `replay_motions.py` | Removed collaborator-specific paths and made model and motion paths explicit; `--list` runs without model assets. |
| `scripts/convert_bones_csv_to_deploy.py` | `gear_sonic_deploy/convert_bones_csv_to_deploy.py` | Fixed headerless CSV detection, added input validation, and clarified that output is for development preview rather than a complete robot deployment package. |

The workflow builds on [NVIDIA NVlabs/GR00T-WholeBodyControl (GEAR-SONIC)](https://github.com/NVlabs/GR00T-WholeBodyControl). The G1 mapping and policy parameters used in the scripts come from that upstream work. Consult the upstream repository for its license and model terms. Robot-side code, model weights, motion data, and raw logs are not included here.
