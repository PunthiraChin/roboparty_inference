from __future__ import annotations

import unittest
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]


class AgentApiContractTests(unittest.TestCase):
    def test_sawasdee_uses_semantic_ids_and_safety_gates(self) -> None:
        with (ROOT / "robots/rpo/configs/sawasdee.yaml").open() as stream:
            params = yaml.safe_load(stream)["inference_node"]["ros__parameters"]
        self.assertEqual(params["policy_ids"], ["locomotion", "sawasdee"])
        self.assertIs(params["hardware_validated"], False)
        self.assertLessEqual(float(params["cmd_vel_timeout_s"]), 0.5)
        self.assertLessEqual(float(params["gravity_z_upper"]), -0.5)

    def test_execute_motion_action_reports_progress(self) -> None:
        action = (ROOT / "action/ExecuteMotion.action").read_text()
        for field in (
            "string motion_id",
            "float32 timeout_s",
            "uint32 frames_executed",
            "uint32 total_frames",
            "float32 progress",
        ):
            self.assertIn(field, action)

    def test_runtime_exposes_ownership_and_freshness(self) -> None:
        state = (ROOT / "msg/RuntimeState.msg").read_text()
        self.assertIn("uint8 command_source", state)
        self.assertIn("bool external_command_fresh", state)
        self.assertIn("string active_policy_id", state)
        self.assertIn("bool hardware_validated", state)
        self.assertIn("bool unvalidated_hardware_override", state)

    def test_generated_interfaces_are_wired_into_build(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text()
        package = (ROOT / "package.xml").read_text()
        self.assertNotIn("-march=native", cmake)
        self.assertIn("rosidl_generate_interfaces", cmake)
        self.assertIn("rclcpp_action", cmake)
        self.assertIn("find_package(action_msgs REQUIRED)", cmake)
        self.assertIn("find_package(rcl_interfaces REQUIRED)", cmake)
        self.assertNotIn("service_msgs", cmake)
        self.assertIn("ament_add_pytest_test(agent_api_contract", cmake)
        self.assertIn("rosidl_default_generators", package)
        self.assertIn("<depend>action_msgs</depend>", package)
        self.assertIn("<depend>rcl_interfaces</depend>", package)
        self.assertNotIn("service_msgs", package)

    def test_package_targets_robot_humble_stack_and_versions_match(self) -> None:
        control = (ROOT / "debian/control").read_text()
        rules = (ROOT / "debian/rules").read_text()
        workflow = (ROOT / ".github/workflows/build-deb.yml").read_text()
        package = (ROOT / "package.xml").read_text()
        changelog = (ROOT / "debian/changelog").read_text()
        for text in (control, rules, workflow):
            self.assertIn("humble", text.lower())
            self.assertNotIn("jazzy", text.lower())
        for dependency in (
            "ccache",
            "libeigen3-dev",
            "ros-humble-action-msgs",
            "ros-humble-rcl-interfaces",
            "ros-humble-std-msgs",
            "ros-humble-sensor-msgs",
            "ros-humble-geometry-msgs",
            "ros-humble-std-srvs",
            "ros-humble-ament-index-python",
            "ros-humble-launch",
            "ros-humble-launch-ros",
            "ros-humble-ros2launch",
        ):
            self.assertIn(dependency, control)
        for dependency in ("ament_index_python", "launch", "launch_ros", "ros2launch"):
            self.assertIn(f"<exec_depend>{dependency}</exec_depend>", package)
        self.assertIn("ubuntu-22.04", workflow)
        self.assertIn("ubuntu-22.04-arm", workflow)
        self.assertNotIn("self-hosted", workflow)
        self.assertNotIn("if: matrix.arch == 'amd64'", workflow)
        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertIn("contents: write", workflow)
        self.assertIn("needs: build", workflow)
        self.assertIn("actions/download-artifact@v4", workflow)
        self.assertIn("gh release download v1.3.0 --repo Roboparty/roboparty_imu", workflow)
        self.assertIn(
            "gh release download v2.2.1 --repo Roboparty/roboparty_motors", workflow
        )
        self.assertIn("sha256sum -c -", workflow)
        for digest in (
            "6ae2ca3d51f88f00182d3cf4ce2052ca6b03d96db677992989d3f147feb50465",
            "2f6390adee13ea2c0ddaaf8cb981b18c3655de6ec3e5032dae26516c8de2e553",
            "59a5f1d9cd54657ebb884b165ba9a4001fa8a6894b04cd7a9a879adf6a9c423b",
            "c566941d62296325f6ef6e8ce6634423f32ca184cf2fba574e922eeb62a5a625",
            "dd98a7fe093ed8145ab843596630555ddd9ebb171a45eaef1ae0d034a37c7412",
        ):
            self.assertIn(digest, workflow)
        self.assertNotIn("Downloading latest", workflow)
        self.assertGreaterEqual(control.count("roboparty-motors (= 2.2.1-1)"), 2)
        self.assertGreaterEqual(control.count("roboparty-imu (= 1.3.0-1)"), 2)
        package_version = re.search(r"<version>([^<]+)</version>", package)
        changelog_version = re.search(r"\(([^-)]+)-", changelog)
        self.assertIsNotNone(package_version)
        self.assertIsNotNone(changelog_version)
        self.assertEqual(package_version.group(1), changelog_version.group(1))

    def test_runtime_has_watchdog_and_no_joystick_injection_service(self) -> None:
        ros_interface = (ROOT / "src/ros_interface.cpp").read_text()
        observations = (ROOT / "src/obs_manager.cpp").read_text()
        self.assertIn("last_external_cmd_time_", ros_interface)
        self.assertIn("cmd_vel_timeout_s_", observations)
        self.assertNotIn("publish<sensor_msgs::msg::Joy>", ros_interface)

    def test_motion_lifecycle_is_serialized_and_acknowledged(self) -> None:
        header = (ROOT / "src/inference_node.hpp").read_text()
        runtime = (ROOT / "src/inference_node.cpp").read_text()
        ros_interface = (ROOT / "src/ros_interface.cpp").read_text()
        self.assertIn("lifecycle_mutex_", header)
        self.assertIn("motion_goal_mutex_", header)
        self.assertIn("active_motion_goal_uuid_", header)
        self.assertIn("applied_action_generation_", header)
        self.assertIn("request_motion_cancel()", ros_interface)
        self.assertIn("robot_->apply_action(filtered_action)", runtime)
        self.assertIn("applied_action_generation_.store", runtime)
        self.assertIn("policy.motion_final_action_generation", ros_interface)

    def test_watchdog_has_a_hard_safe_range_and_locomotion_is_semantic(self) -> None:
        ros_interface = (ROOT / "src/ros_interface.cpp").read_text()
        self.assertIn("cmd_vel_timeout_s_ > 0.5", ros_interface)
        self.assertIn('find_policy_by_id("locomotion")', ros_interface)
        self.assertNotIn("active_policy_idx_ = 0", ros_interface)
        self.assertIn(": locomotion_policy_idx_", ros_interface)

    def test_cancel_is_checked_before_motion_activation(self) -> None:
        runtime = (ROOT / "src/inference_node.cpp").read_text()
        ros_interface = (ROOT / "src/ros_interface.cpp").read_text()
        self.assertIn("active_motion_goal_uuid_ != goal_uuid", runtime)
        self.assertIn("motion_cancel_requested_.load()", runtime)
        self.assertIn("goal_handle->get_goal_id()", ros_interface)

    def test_command_source_is_idempotent_and_runtime_state_is_truthful(self) -> None:
        header = (ROOT / "src/inference_node.hpp").read_text()
        runtime = (ROOT / "src/inference_node.cpp").read_text()
        observations = (ROOT / "src/obs_manager.cpp").read_text()
        ros_interface = (ROOT / "src/ros_interface.cpp").read_text()
        self.assertIn("already selected", ros_interface)
        self.assertIn("motion_action_active_.load() || is_motion_policy_.load()", ros_interface)
        self.assertIn("runtime_fault_mutex_", header)
        self.assertIn("set_runtime_fault", runtime)
        self.assertIn("robot_fall_detected", observations)
        cmd_callback = ros_interface.split("void InferenceNode::subs_cmd_callback", 1)[1]
        cmd_callback = cmd_callback.split("void InferenceNode::subs_perception_callback", 1)[0]
        self.assertIn("lifecycle_mutex_", cmd_callback)
        self.assertLess(
            cmd_callback.index("lifecycle_mutex_"), cmd_callback.index("if(!is_joy_control_)")
        )

    def test_worker_faults_are_published_before_shutdown(self) -> None:
        header = (ROOT / "src/inference_node.hpp").read_text()
        runtime = (ROOT / "src/inference_node.cpp").read_text()
        ros_interface = (ROOT / "src/ros_interface.cpp").read_text()
        publisher_position = header.index("runtime_state_publisher_ =")
        worker_position = header.index("inference_thread_ =")
        self.assertLess(publisher_position, worker_position)
        self.assertIn("publish_terminal_fault_state(fault)", runtime)
        self.assertIn("state.inference_running = false", ros_interface)
        self.assertIn("state.fault = fault", ros_interface)

    def test_external_inputs_and_policy_outputs_reject_nonfinite_values(self) -> None:
        runtime = (ROOT / "src/inference_node.cpp").read_text()
        ros_interface = (ROOT / "src/ros_interface.cpp").read_text()
        self.assertIn("nonfinite_joy_input", ros_interface)
        self.assertIn("nonfinite_cmd_vel_input", ros_interface)
        self.assertIn("nonfinite_perception_input", ros_interface)
        self.assertIn("malformed_interrupt_joint_state", ros_interface)
        self.assertIn("nonfinite_interrupt_joint_state", ros_interface)
        self.assertIn("non_finite_policy_observation", runtime)
        self.assertIn("non_finite_policy_input", runtime)
        self.assertIn("non_finite_policy_output", runtime)
        self.assertIn("non_finite_action_target", runtime)
        self.assertIn("action_target_outside_joint_limits", runtime)

    def test_sawasdee_uses_canonical_hardware_joint_limits(self) -> None:
        with (ROOT / "robots/rpo/configs/default.yaml").open() as stream:
            default = yaml.safe_load(stream)["inference_node"]["ros__parameters"]
        with (ROOT / "robots/rpo/configs/sawasdee.yaml").open() as stream:
            sawasdee = yaml.safe_load(stream)["inference_node"]["ros__parameters"]
        with (ROOT / "robots/rpo/sawasdee.manifest.yaml").open() as stream:
            manifest = yaml.safe_load(stream)
        validator = (ROOT / "tools/validate_motion_policy.py").read_text()
        self.assertEqual(sawasdee["joint_limits"], default["joint_limits"])
        self.assertIs(sawasdee["hardware_validated"], False)
        self.assertIs(manifest["runtime"]["hardware_validated"], False)
        self.assertEqual(manifest["runtime"]["validation_status"], "offline_and_mujoco_only")
        self.assertIn("canonical RPO hardware joint limits", validator)
        self.assertIn("hardware: BLOCKED (offline/simulation validation only)", validator)

    def test_unvalidated_profiles_fail_closed_without_blocking_shutdown(self) -> None:
        header = (ROOT / "src/inference_node.hpp").read_text()
        runtime = (ROOT / "src/inference_node.cpp").read_text()
        ros_interface = (ROOT / "src/ros_interface.cpp").read_text()
        launch = (ROOT / "launch/inference.launch.py").read_text()

        self.assertIn("bool hardware_validated_ = false", header)
        self.assertIn("bool allow_unvalidated_hardware_ = false", header)
        self.assertIn("hardware_execution_allowed", header)
        self.assertIn('declare_parameter<bool>("hardware_validated", false', ros_interface)
        self.assertIn("hardware_validated_descriptor.read_only = true", ros_interface)
        self.assertIn("hardware_override_descriptor.read_only = true", ros_interface)
        self.assertIn('"allow_unvalidated_hardware", false', ros_interface)
        self.assertIn("if (!hardware_execution_allowed(blocked_reason))", ros_interface)
        self.assertIn("if (!hardware_execution_allowed(reason))", runtime)
        self.assertIn('DeclareLaunchArgument("allow_unvalidated_hardware", default_value="false")', launch)

        established_profiles = (
            "default.yaml",
            "amp.yaml",
            "attn_enc.yaml",
            "beyondmimic.yaml",
            "getup.yaml",
            "interrupt.yaml",
            "parkour.yaml",
        )
        for filename in established_profiles:
            with self.subTest(filename=filename):
                with (ROOT / "robots/rpo/configs" / filename).open() as stream:
                    params = yaml.safe_load(stream)["inference_node"]["ros__parameters"]
                self.assertIs(params["hardware_validated"], True)

        deinit = ros_interface.split("void InferenceNode::deinit_motors_srv", 1)[1]
        deinit = deinit.split("void InferenceNode::start_inference_srv", 1)[0]
        self.assertNotIn("hardware_execution_allowed", deinit)
        stop = ros_interface.split("void InferenceNode::stop_inference_srv", 1)[1]
        stop = stop.split("void InferenceNode::set_command_source_srv", 1)[0]
        self.assertNotIn("hardware_execution_allowed", stop)

    def test_config_and_onnx_contracts_are_validated_before_workers_start(self) -> None:
        header = (ROOT / "src/inference_node.hpp").read_text()
        runtime = (ROOT / "src/inference_node.cpp").read_text()
        ros_interface = (ROOT / "src/ros_interface.cpp").read_text()
        validator = (ROOT / "tools/validate_motion_policy.py").read_text()
        self.assertIn("usd2urdf must be a permutation", ros_interface)
        self.assertIn("joint_limits must contain two finite values per joint", ros_interface)
        self.assertIn("clip_cmd must contain three finite lower/upper pairs", ros_interface)
        self.assertIn("Only single-output ONNX models are supported", runtime)
        self.assertIn("ONNX output tensor must have shape [1, joint_num]", runtime)
        self.assertIn("len(inputs) != 1 or len(outputs) != 1", validator)
        self.assertLess(header.index("load_config();"), header.index("inference_thread_ ="))

    def test_latched_fault_or_active_transition_cannot_resume_inference(self) -> None:
        runtime = (ROOT / "src/inference_node.cpp").read_text()
        ros_interface = (ROOT / "src/ros_interface.cpp").read_text()
        self.assertIn("bool InferenceNode::try_start_inference", runtime)
        self.assertIn("Runtime fault is latched", runtime)
        self.assertIn("motion_action_active_.load()", runtime)
        self.assertIn("reset_thread_running_", runtime)
        self.assertIn("std::unique_lock<std::mutex> fault_lock(runtime_fault_mutex_)", runtime)
        self.assertIn("is_running_.store(true)", runtime)
        self.assertGreaterEqual(ros_interface.count("try_start_inference(blocked_reason)"), 2)
        self.assertIn("resume_inference_if_fault_free", runtime)
        self.assertIn("resume_inference_if_fault_free", ros_interface)
        self.assertEqual(runtime.count("is_running_.store(true)"), 1)
        self.assertNotIn("is_running_.store(true)", ros_interface)

    def test_motor_initialization_fails_closed_and_rolls_back(self) -> None:
        header = (ROOT / "include/robot_interface.hpp").read_text()
        robot = (ROOT / "src/robot_interface.cpp").read_text()
        init = robot.split("void RobotInterface::init_motors()", 1)[1]
        init = init.split("void RobotInterface::deinit_motors()", 1)[0]
        self.assertIn("rollback_motor_initialization() noexcept", header)
        self.assertIn("init_status[idx] = motor->init_motor()", init)
        self.assertIn("response_count[idx] = motor->get_response_count()", init)
        self.assertIn("init_status[idx] == 0 && response_count[idx] == 0", init)
        self.assertIn("rollback_motor_initialization();", init)
        self.assertLess(init.index("rollback_motor_initialization();"), init.index("is_init_.store(true)"))
        rollback = init.split("void RobotInterface::rollback_motor_initialization()", 1)[1]
        self.assertIn("is_init_.store(false)", rollback)
        self.assertIn("motor->deinit_motor()", rollback)
        self.assertIn("rollback_errors[idx] = std::current_exception()", rollback)


if __name__ == "__main__":
    unittest.main()
