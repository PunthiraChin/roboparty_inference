# Sawasdee policy

This integration adds the trained knee-bend Sawasdee motion as a second policy
beside RoboParty's stock locomotion policy. It reuses the existing BeyondMimic
action-to-motor interface; the motor-driver repositories are unchanged.

The Debian package and CI target the robot deployment baseline: Ubuntu 22.04
with ROS 2 Humble, on both amd64 and arm64.

## Runtime contract

- Policy input: `[1, 121]`
- Policy output: `[1, 23]` joint actions
- Motion reference: 383 frames, 23 joints, 50 Hz
- Control rate: `dt=0.004`, `decimation=5`, therefore 50 Hz
- Policy action scale: 0.25
- Joint order: the existing `usd2urdf` mapping in `sawasdee.yaml`

## Validate without robot hardware

From the `roboparty_inference` repository:

```bash
python3 tools/validate_motion_policy.py
```

This checks the YAML wiring, canonical joint limits, hardware-validation gate,
ONNX input/output contract, motion arrays, hashes, and one CPU inference. It
does not initialize or contact motors.

Artifact provenance and expected hashes are recorded in
`sawasdee.manifest.yaml`.

## Physical-hardware gate

The bundled Sawasdee profile is **offline/simulation only**. It is deliberately
configured with `hardware_validated: false`, so the runtime blocks motor
initialization, inference start, and motion goals by default. Preflight replay
found that the exported policy can request targets outside RoboParty's
canonical joint envelope under plausible tracking states. Passing MuJoCo is a
useful transfer check, but is not physical-hardware validation.

Do not use the normal robot start flow for this profile yet. Before changing
the flag to `true`, retrain or safety-tune the policy to respect the canonical
limits, rerun the offline/MuJoCo regressions, then complete supervised tests on
a physical stand with a working emergency stop and progressively bounded
motion. Record that evidence in the policy manifest and review it independently.

An explicit startup-only `allow_unvalidated_hardware` override exists for a
controlled engineering validation session. It is read-only after startup,
published in `/runtime_state`, and must never be used for normal operation.

After hardware validation, the normal manual flow is:

1. Put the robot on a supported stand and clear the surrounding area.
2. Use `X` for the normal motor initialization flow.
3. Use `A` to reset to the default joint pose.
4. Use `B` to start inference.
5. Use `LB` to toggle between stock locomotion and Sawasdee.
6. Use `B` to pause, then deinitialize safely with `X` when finished.

The motion is not a loop. The typed action advances through its 383 reference
frames, reports progress, and then automatically returns to locomotion.
Cancellation, timeout, joystick override, inference stop, and motor
deinitialization also cancel the action and zero the velocity buffer.

## Typed agent API (after hardware validation)

The action accepts semantic IDs, never numeric policy indexes:

```bash
ros2 action send_goal --feedback \
  /execute_motion roboparty_inference/action/ExecuteMotion \
  "{motion_id: sawasdee, timeout_s: 12.0}"
```

Select the sole external velocity source explicitly:

```bash
ros2 service call /set_command_source \
  roboparty_inference/srv/SetCommandSource "{source: 1}"
```

`/runtime_state` reports command ownership, active policy, action/manual motion
activity, motion progress, the first latched runtime fault, and whether the
latest external command is fresh. Re-selecting the current command source is
idempotent. External `/cmd_vel` expires after 250 ms by default and becomes
zero inside the inference process. A malformed/non-finite sensor or command
message, non-finite policy value, or out-of-limit action latches a terminal
fault; inference cannot be resumed until the node is restarted. A cancelling
motion or active joint reset also blocks resume.

These interfaces must be tested in a ROS-only/offline harness and simulation
before they are exercised on physical hardware. With the current Sawasdee
bundle, the hardware gate intentionally rejects the action.

## Safety boundary for a future VLA agent

A VLA must request a named skill such as `sawasdee`; it must never publish raw
joint commands. A deterministic Skill Manager should own action lifecycle and
velocity publication. A Safety Supervisor and human stop always take priority.
