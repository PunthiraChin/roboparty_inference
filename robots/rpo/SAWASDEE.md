# Sawasdee policy

This integration adds the trained knee-bend Sawasdee motion as a second policy
beside RoboParty's stock locomotion policy. It reuses the existing BeyondMimic
runtime; no motor-control code is changed.

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

This checks the YAML wiring, ONNX input/output contract, motion arrays, hashes,
and one CPU inference. It does not initialize or contact motors.

Artifact provenance and expected hashes are recorded in
`sawasdee.manifest.yaml`.

## Start on RoboParty

Build and source the ROS 2 workspace using the normal deployment instructions,
then select the new configuration:

```bash
./tools/start_robot.sh --robot rpo --policy sawasdee
```

The command is run from the top-level `roboparty_deploy` repository, not from
this submodule.

Current gamepad flow:

1. Put the robot on a supported stand and clear the surrounding area.
2. Use `X` for the normal motor initialization flow.
3. Use `A` to reset to the default joint pose.
4. Use `B` to start inference.
5. Use `LB` to toggle between stock locomotion and Sawasdee.
6. Use `B` to pause, then deinitialize safely with `X` when finished.

The motion is not a loop. The runtime advances through its 383 reference frames
and clamps at the final frame. Return-to-stand and repeat behavior should be
implemented as an explicit skill lifecycle before an autonomous agent invokes
the policy.

## Safety boundary for a future VLA agent

A VLA must request a named skill such as `sawasdee`; it must never publish raw
joint commands. A deterministic Skill Manager should own policy selection,
reset the motion frame, supervise completion or timeout, and return to the
standing controller. A Safety Supervisor and human stop always take priority.
