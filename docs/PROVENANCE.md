# Code provenance and attribution

This repository publishes three selected **development-machine** scripts from the 2026 World Humanoid Robot Games handover. Haojin Lu confirmed that 超能逸仙队 permits public release of these specific files. The archived scripts were untracked in the supplied development snapshot, so file-level authorship cannot be recovered reliably from Git history. They are attributed to the **competition team**; this repository does not claim Haojin Lu wrote each line or authored the underlying policy.

| Public file | Source in the handover | Public-release changes |
| --- | --- | --- |
| `scripts/run_policy_sim.py` | `run_policy_sim.py` | Removed collaborator-specific paths and made model, ONNX and motion paths explicit; `--list` runs without model assets. |
| `scripts/replay_motions.py` | `replay_motions.py` | Removed collaborator-specific paths and made model and motion paths explicit; `--list` runs without model assets. |
| `scripts/convert_bones_csv_to_deploy.py` | `gear_sonic_deploy/convert_bones_csv_to_deploy.py` | Fixed headerless CSV detection, added input validation, and clarified that output is for development preview rather than a complete robot deployment package. |

The broader system builds on [NVIDIA NVlabs/GR00T-WholeBodyControl (GEAR-SONIC)](https://github.com/NVlabs/GR00T-WholeBodyControl). Its framework, pretrained models, joint mapping and policy parameters remain credited to their upstream authors. The legal bundle in the supplied handover identifies an Apache 2.0 notice for GEAR-SONIC; consult the [upstream repository](https://github.com/NVlabs/GR00T-WholeBodyControl) and its current legal files for reuse terms. That upstream notice does not assert sole ownership of team-written code or grant a blanket license for the full competition system. Model weights are not included and may have separate terms.

Not released: robot-side code, model weights, motion assets, raw logs, operator notes, network settings, credentials, bundled upstream/third-party repositories, and unpublished HRI research materials. The three scripts are development utilities, not a reproducible competition stack or evidence of any measured performance.
