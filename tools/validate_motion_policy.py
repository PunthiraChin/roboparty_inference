#!/usr/bin/env python3
"""Validate a RoboParty motion policy bundle without contacting robot hardware."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layout_size(layout: str) -> int:
    return sum(int(field.rsplit(":", 1)[1]) for field in layout.split(", "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="roboparty_inference repository root",
    )
    args = parser.parse_args()

    rpo = args.repo / "robots" / "rpo"
    config_path = rpo / "configs" / "sawasdee.yaml"
    manifest_path = rpo / "sawasdee.manifest.yaml"
    with config_path.open() as stream:
        params = yaml.safe_load(stream)["inference_node"]["ros__parameters"]
    with manifest_path.open() as stream:
        manifest = yaml.safe_load(stream)

    names = params["model_names"]
    motions = params["motion_names"]
    layouts = params["obs_layouts"]
    stacks = params["frame_stacks"]
    if not (len(names) == len(motions) == len(layouts) == len(stacks) == 2):
        raise ValueError("Sawasdee config must contain exactly locomotion + motion entries")

    model_path = rpo / "models" / names[1]
    motion_path = rpo / "motions" / motions[1]
    expected_input = layout_size(layouts[1]) * int(stacks[1])
    if model_path != rpo / manifest["policy"]["path"]:
        raise ValueError("Config and manifest refer to different policy files")
    if motion_path != rpo / manifest["motion"]["path"]:
        raise ValueError("Config and manifest refer to different motion files")
    if sha256(model_path) != manifest["policy"]["sha256"]:
        raise ValueError("Policy SHA-256 does not match its manifest")
    if sha256(motion_path) != manifest["motion"]["sha256"]:
        raise ValueError("Motion SHA-256 does not match its manifest")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.shape != [1, expected_input]:
        raise ValueError(f"ONNX input {input_meta.shape}; expected [1, {expected_input}]")
    if output_meta.shape != [1, int(params["joint_num"])]:
        raise ValueError(f"ONNX output {output_meta.shape}; expected [1, 23]")
    output = session.run(None, {input_meta.name: np.zeros((1, expected_input), np.float32)})[0]
    if output.shape != (1, 23) or not np.isfinite(output).all():
        raise ValueError("ONNX smoke inference returned invalid output")

    with np.load(motion_path) as motion:
        required = {"fps", "joint_pos", "joint_vel"}
        missing = required.difference(motion.files)
        if missing:
            raise ValueError(f"Motion is missing arrays: {sorted(missing)}")
        joint_pos = motion["joint_pos"]
        joint_vel = motion["joint_vel"]
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        if joint_pos.ndim != 2 or joint_pos.shape[1] != 23:
            raise ValueError(f"joint_pos shape is {joint_pos.shape}; expected [frames, 23]")
        if joint_vel.shape != joint_pos.shape:
            raise ValueError("joint_vel must match joint_pos")
        if joint_pos.dtype != np.float32 or joint_vel.dtype != np.float32:
            raise ValueError("joint_pos and joint_vel must be float32")
        if not np.isclose(fps, 50.0):
            raise ValueError(f"Motion is {fps} Hz; expected 50 Hz")

    control_hz = 1.0 / (float(params["dt"]) * int(params["decimation"]))
    if not np.isclose(control_hz, fps):
        raise ValueError(f"Controller is {control_hz} Hz but motion is {fps} Hz")

    print("Sawasdee bundle: VALID")
    print(f"  policy: {model_path.name}  sha256={sha256(model_path)}")
    print(f"  motion: {motion_path.name}  sha256={sha256(motion_path)}")
    print(f"  contract: [1, {expected_input}] -> [1, 23]")
    print(f"  motion: {joint_pos.shape[0]} frames at {fps:g} Hz ({joint_pos.shape[0] / fps:.2f}s)")
    print(f"  control: {control_hz:g} Hz")


if __name__ == "__main__":
    main()
