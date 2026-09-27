# Technical scope

This page records the engineering workflow visible in the competition handover. It is deliberately high level: implementation details, deployment configuration, models, motion files, and experiment logs are not released.

## Development and deployment workflow

```text
Motion assets
    ↓
Workstation-side preparation and MuJoCo checks
    ↓
Onboard policy and reference-motion deployment
    ↓
Unitree G1 testing and debugging
    ↺
```

The handover includes a headless MuJoCo smoke-check workflow, ONNX inference/deployment references, a reference-motion library, and separate development-machine and robot-side control paths. It also documents operator input modes for simulation and the physical robot. The two contexts require different interfaces and validation; this is why integration and real-robot testing were central to the work.

These observations describe the archived system, not exclusive authorship. The current public repository includes no executable code. It does not establish the performance or reliability of an individual component, and it does not imply the public can reproduce the competition setup from this page.

## Engineering concerns visible in the archive

- **Motion compatibility:** the local replay and simulation scripts handle the G1 joint ordering and motion frame rate explicitly. Kinematic replay provides a way to inspect motion data before testing it through the full controller.
- **Simulation checks:** a headless MuJoCo path combines reference motions with ONNX encoder and decoder inference. Its presence supports a workstation-side check workflow; it is not a published benchmark or evidence of a particular success rate.
- **Motion preparation:** the archived Tai Chi scripts explore trajectory variants, retiming, and reset checks. Their presence does not establish that every generated variant was used in competition or that I authored every script.
- **Robot deployment:** the development copy includes reference-motion playback and gamepad input paths. The separate robot-side copy has not yet been inspected or compared with this development copy.

These are observations from the development archive. No numerical result, safety guarantee, trained-model contribution, or fully reproducible system is claimed.

## My role

As Team Lead / Technical Integration for 超能逸仙队, I coordinated the technical workflow and contributed to G1 motion integration, deployment, and real-robot testing and debugging. Work by teammates, the base framework, third-party dependencies, and model authors remains their own.
