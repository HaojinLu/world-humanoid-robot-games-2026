# World Humanoid Robot Games 2026 — Unitree G1

A public overview of my work with **超能逸仙队** at the 2026 World Humanoid Robot Games. I served as **Team Lead / Technical Integration** for the team's Unitree G1 participation in Street Dance and Tai Chi. This was a team effort; the results below belong to the team.

## Competition results

| Event | Team result |
| --- | --- |
| Street Dance | Top 16 (team-reported) |
| Tai Chi | [11th place](https://robopodium.com/whrg-2026/wushu/taijiquan) |

## My role

- Team coordination and technical integration across development and robot deployment.
- Integration of whole-body motion workflows for the Unitree G1.
- Real-robot testing, debugging and deployment support.

The team built on [NVIDIA NVlabs/GR00T-WholeBodyControl (GEAR-SONIC)](https://github.com/NVlabs/GR00T-WholeBodyControl). I do not claim to have designed or trained its underlying policy. See [code provenance and attribution](docs/PROVENANCE.md) and [technical scope](docs/TECHNICAL_SCOPE.md).

## Selected development-machine code

This repository includes three reviewed team scripts for motion preparation, MuJoCo kinematic replay and ONNX policy simulation. They have been curated to remove collaborator-specific paths and to require externally supplied models and motion data. The public repository does **not** contain those assets or the robot-side implementation.

| Script | Purpose |
| --- | --- |
| [`convert_bones_csv_to_deploy.py`](scripts/convert_bones_csv_to_deploy.py) | Resample a Bones-style CSV to four 50 Hz development-preview CSVs and metadata. Despite its historical filename, this is **not** a complete deployment-data generator. |
| [`replay_motions.py`](scripts/replay_motions.py) | Kinematic motion preview in a user-supplied G1 MuJoCo model. |
| [`run_policy_sim.py`](scripts/run_policy_sim.py) | MuJoCo physics check with user-supplied G1 model and ONNX encoder/decoder; `--no-policy` enables a kinematic comparison. |

### Local checks

Use Python 3.11 or newer. For the converter and tests:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/convert_bones_csv_to_deploy.py --input PATH_TO_BONES_CSV_DIR --output PATH_TO_PREVIEW_DIR --fps-source 120
```

For simulation, additionally install `requirements-sim.txt` and supply a G1 MuJoCo XML, motion CSV directory and (for policy mode) the ONNX models **obtained separately under their applicable terms**:

```bash
python -m pip install -r requirements-sim.txt
python scripts/replay_motions.py --motion-dir PATH_TO_MOTIONS --list
python scripts/replay_motions.py --model-xml PATH_TO_G1_XML --motion-dir PATH_TO_MOTIONS
python scripts/run_policy_sim.py --motion-dir PATH_TO_MOTIONS --list
python scripts/run_policy_sim.py --model-xml PATH_TO_G1_XML --motion-dir PATH_TO_MOTIONS --no-policy --headless --sim-seconds 2
python scripts/run_policy_sim.py --model-xml PATH_TO_G1_XML --motion-dir PATH_TO_MOTIONS --encoder PATH_TO_ENCODER_ONNX --decoder PATH_TO_DECODER_ONNX --headless --sim-seconds 2
```

The converter has been checked with generated headered and headerless CSVs. Script syntax and command-line help were checked. **Full simulation and real-robot execution have not been validated from this public checkout**, because models, motion assets and the robot-side stack are intentionally absent.

## Release boundary

No credentials, network settings, raw robot logs, model weights, motion trajectories, bundled third-party repository, or robot-side code are published. Robot-side code may be added later after a separate review. No unpublished HRI research code, participant data, study material, manuscript figure or detailed result is included. This repository is a selected engineering record, not the full competition system or a performance benchmark.
