# World Humanoid Robot Games 2026 — Unitree G1

A public overview of my work with **超能逸仙队** at the 2026 World Humanoid Robot Games. I served as **Team Lead / Technical Integration** for the team's Unitree G1 participation in Street Dance and Tai Chi.

This is a team competition project. The results below belong to the team, and the technical description distinguishes my integration role from the broader work of teammates and upstream software authors.

## Competition results

| Event | Team result |
| --- | --- |
| Street Dance | Top 16 (team-reported) |
| Tai Chi | [11th place](https://robopodium.com/whrg-2026/wushu/taijiquan) |

## Technical scope

The development workflow connected motion assets, workstation-side simulation and checks, onboard deployment, and testing on the G1. The available handover materials document MuJoCo simulation, ONNX-based policy deployment, reference-motion playback, and robot-side control interfaces. These components were integrated and tested as a system for the competition; this repository does not claim that I designed or trained the underlying policy.

My contribution areas were:

- Team coordination and technical integration across development and robot deployment.
- Integration of whole-body motion workflows for the Unitree G1.
- Real-robot testing, debugging, and deployment support.

A concise description of the visible engineering workflow is in [Technical scope](docs/TECHNICAL_SCOPE.md).

## Code and materials

This repository is a **curated public overview**, not a release of the full competition stack. The supplied archive combines team work, upstream code, third-party dependencies, model files, motion assets, and experiment logs. Its provenance and release boundaries need to be checked file by file before any source can be published. Robot-side files outside that archive have not been reviewed here.

No credentials, internal network settings, raw robot logs, model weights, motion trajectories, or bundled third-party code are included. There is no runnable public implementation or reproducibility claim in this version.

## Attribution and research boundaries

The system uses existing whole-body-control software and other third-party components. I claim my team leadership and integration contribution, not sole authorship of the full system, pretrained models, or all competition assets.

No unpublished research code, study material, participant data, manuscript content, or results are included here.
