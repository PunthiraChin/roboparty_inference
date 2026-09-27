// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "inference_node.hpp"

#include <limits>

void InferenceNode::load_config() {
    const std::string default_robot_dir = std::string(ROOT_DIR) + "robots/rpo";
    this->declare_parameter<std::string>("robot_name", "rpo");
    this->declare_parameter<std::string>("policy_name", "default");
    this->declare_parameter<std::string>("robot_config", default_robot_dir + "/robot.yaml");
    this->declare_parameter<std::string>("model_dir", default_robot_dir + "/models");
    this->declare_parameter<std::string>("motion_dir", default_robot_dir + "/motions");
    this->declare_parameter<std::string>("latent_dir", default_robot_dir + "/latents");
    this->declare_parameter<std::vector<std::string>>("model_names", std::vector<std::string>{});
    this->declare_parameter<std::vector<std::string>>("policy_ids", std::vector<std::string>{});
    this->declare_parameter<std::vector<std::string>>("motion_names", std::vector<std::string>{});
    this->declare_parameter<std::vector<std::string>>("latent_names", std::vector<std::string>{});
    this->declare_parameter<std::vector<std::string>>("obs_layouts", std::vector<std::string>{});
    this->declare_parameter<std::vector<long int>>("frame_stacks", std::vector<long int>{});
    this->declare_parameter<std::vector<std::string>>("obs_stack_orders", std::vector<std::string>{});
    this->declare_parameter<float>("act_alpha", 0.9);
    this->declare_parameter<int>("intra_threads", -1);
    this->declare_parameter<std::string>("perception_obs_topic", "elevation_data");
    this->declare_parameter<bool>("use_depth", false);
    this->declare_parameter<int>("joint_num", 23);
    this->declare_parameter<int>("decimation", 10);
    this->declare_parameter<float>("dt", 0.001);
    this->declare_parameter<float>("obs_scales_lin_vel", 1.0);
    this->declare_parameter<float>("obs_scales_ang_vel", 1.0);
    this->declare_parameter<float>("obs_scales_dof_pos", 1.0);
    this->declare_parameter<float>("obs_scales_dof_vel", 1.0);
    this->declare_parameter<float>("obs_scales_gravity_b", 1.0);
    this->declare_parameter<float>("clip_observations", 100.0);
    this->declare_parameter<std::vector<double>>("action_scale", std::vector<double>{0.3});
    this->declare_parameter<float>("clip_actions", 18.0);
    this->declare_parameter<double>("action_rescale", 1.0);
    this->declare_parameter<std::vector<long int>>("usd2urdf", std::vector<long int>{});
    this->declare_parameter<std::vector<double>>(
        "clip_cmd", std::vector<double>{-0.4, 0.6, -0.4, 0.4, -0.8, 0.8});
    this->declare_parameter<double>("cmd_vel_timeout_s", 0.25);
    this->declare_parameter<std::vector<double>>("joint_default_angle", std::vector<double>{});
    this->declare_parameter<std::vector<double>>("joint_limits", std::vector<double>{});
    this->declare_parameter<float>("gravity_z_upper", -0.5);
    this->declare_parameter<double>("gamma", 0.8);
    this->declare_parameter<int>("window_size", 1);
    std::vector<std::string> model_names;
    std::vector<std::string> policy_ids;
    std::vector<std::string> motion_names;
    std::vector<std::string> latent_names;
    std::vector<std::string> obs_layouts;
    std::vector<long int> frame_stacks;
    std::vector<std::string> obs_stack_orders;
    std::string robot_name;
    std::string policy_name;
    std::string model_dir;
    std::string motion_dir;
    std::string latent_dir;
    this->get_parameter("robot_name", robot_name);
    this->get_parameter("policy_name", policy_name);
    this->get_parameter("robot_config", robot_config_path_);
    this->get_parameter("model_dir", model_dir);
    this->get_parameter("motion_dir", motion_dir);
    this->get_parameter("latent_dir", latent_dir);
    this->get_parameter("model_names", model_names);
    this->get_parameter("policy_ids", policy_ids);
    this->get_parameter("motion_names", motion_names);
    this->get_parameter("latent_names", latent_names);
    this->get_parameter("obs_layouts", obs_layouts);
    this->get_parameter("frame_stacks", frame_stacks);
    this->get_parameter("obs_stack_orders", obs_stack_orders);
    this->get_parameter("act_alpha", act_alpha_);
    this->get_parameter("intra_threads", intra_threads_);
    this->get_parameter("perception_obs_topic", perception_obs_topic_);
    this->get_parameter("use_depth", use_depth_);
    this->get_parameter("joint_num", joint_num_);
    this->get_parameter("decimation", decimation_);
    this->get_parameter("dt", dt_);
    this->get_parameter("obs_scales_lin_vel", obs_scales_lin_vel_);
    this->get_parameter("obs_scales_ang_vel", obs_scales_ang_vel_);
    this->get_parameter("obs_scales_dof_pos", obs_scales_dof_pos_);
    this->get_parameter("obs_scales_dof_vel", obs_scales_dof_vel_);
    this->get_parameter("obs_scales_gravity_b", obs_scales_gravity_b_);
    this->get_parameter("clip_observations", clip_observations_);
    this->get_parameter("action_scale", action_scale_);
    double action_rescale = 1.0;
    this->get_parameter("action_rescale", action_rescale);
    action_rescale_ = static_cast<float>(action_rescale);
    if (joint_num_ <= 0) {
        throw std::runtime_error("joint_num must be greater than zero");
    }
    if (action_scale_.size() == 1) {
        action_scale_.resize(static_cast<std::size_t>(joint_num_), action_scale_.front());
    } else if (action_scale_.size() != static_cast<std::size_t>(joint_num_)) {
        throw std::runtime_error(
            "action_scale must contain either 1 value or " +
            std::to_string(joint_num_) + " values, but got " +
            std::to_string(action_scale_.size()));
    }
    this->get_parameter("clip_actions", clip_actions_);
    this->get_parameter("usd2urdf", usd2urdf_);
    this->get_parameter("clip_cmd", clip_cmd_);
    this->get_parameter("cmd_vel_timeout_s", cmd_vel_timeout_s_);
    if (!std::isfinite(cmd_vel_timeout_s_) ||
        cmd_vel_timeout_s_ < 0.05 || cmd_vel_timeout_s_ > 0.5) {
        throw std::runtime_error("cmd_vel_timeout_s must be between 0.05 and 0.5 seconds");
    }
    this->get_parameter("joint_default_angle", joint_default_angle_);
    this->get_parameter("joint_limits", joint_limits_);
    this->get_parameter("gravity_z_upper", gravity_z_upper_);
    double latent_gamma = 0.8;
    this->get_parameter("gamma", latent_gamma);
    int latent_window_size = 1;
    this->get_parameter("window_size", latent_window_size);

    const auto all_finite = [](const auto& values) {
        return std::all_of(values.begin(), values.end(), [](const auto value) {
            return std::isfinite(static_cast<double>(value));
        });
    };
    if (!std::isfinite(act_alpha_) || act_alpha_ < 0.0f || act_alpha_ > 1.0f) {
        throw std::runtime_error("act_alpha must be finite and between 0 and 1");
    }
    if (!std::isfinite(dt_) || dt_ < 0.000001f || decimation_ <= 0) {
        throw std::runtime_error("dt and decimation must define a positive control period");
    }
    if (!std::isfinite(clip_observations_) || clip_observations_ <= 0.0f ||
        !std::isfinite(clip_actions_) || clip_actions_ <= 0.0f ||
        !std::isfinite(action_rescale_) || !std::isfinite(gravity_z_upper_)) {
        throw std::runtime_error("policy scaling and safety parameters must be finite");
    }
    if (!all_finite(action_scale_)) {
        throw std::runtime_error("action_scale must contain only finite values");
    }
    if (clip_cmd_.size() != 6 || !all_finite(clip_cmd_) ||
        clip_cmd_[0] > clip_cmd_[1] || clip_cmd_[2] > clip_cmd_[3] ||
        clip_cmd_[4] > clip_cmd_[5]) {
        throw std::runtime_error("clip_cmd must contain three finite lower/upper pairs");
    }
    if (joint_default_angle_.size() != static_cast<size_t>(joint_num_) ||
        !all_finite(joint_default_angle_)) {
        throw std::runtime_error("joint_default_angle must contain one finite value per joint");
    }
    if (joint_limits_.size() != static_cast<size_t>(joint_num_) * 2 ||
        !all_finite(joint_limits_)) {
        throw std::runtime_error("joint_limits must contain two finite values per joint");
    }
    for (int joint = 0; joint < joint_num_; joint++) {
        const size_t offset = static_cast<size_t>(joint) * 2;
        if (joint_limits_[offset] > joint_limits_[offset + 1] ||
            joint_default_angle_[joint] < joint_limits_[offset] ||
            joint_default_angle_[joint] > joint_limits_[offset + 1]) {
            throw std::runtime_error("joint limits must contain the default joint position");
        }
    }
    if (usd2urdf_.size() != static_cast<size_t>(joint_num_)) {
        throw std::runtime_error("usd2urdf must contain one index per joint");
    }
    std::vector<bool> mapped_joints(static_cast<size_t>(joint_num_), false);
    for (const long int index : usd2urdf_) {
        if (index < 0 || index >= joint_num_ || mapped_joints[static_cast<size_t>(index)]) {
            throw std::runtime_error("usd2urdf must be a permutation of all joint indices");
        }
        mapped_joints[static_cast<size_t>(index)] = true;
    }

    policies_.clear();
    motion_policy_indices_.clear();
    perception_obs_num_ = 0;
    const size_t policy_count = model_names.size();
    if (policy_count == 0) {
        throw std::runtime_error("model_names must contain at least one policy");
    }
    const auto require_policy_count = [policy_count](const auto& values, const std::string& name) {
        if (values.size() != policy_count) {
            throw std::runtime_error(name + " must have the same size as model_names");
        }
    };
    const auto require_empty_or_policy_count = [policy_count](const auto& values, const std::string& name) {
        if (!values.empty() && values.size() != policy_count) {
            throw std::runtime_error(name + " must be empty or have the same size as model_names");
        }
    };
    require_policy_count(obs_layouts, "obs_layouts");
    require_policy_count(frame_stacks, "frame_stacks");
    require_policy_count(obs_stack_orders, "obs_stack_orders");
    require_empty_or_policy_count(motion_names, "motion_names");
    require_empty_or_policy_count(latent_names, "latent_names");
    require_empty_or_policy_count(policy_ids, "policy_ids");

    const auto resolve_asset_path = [](const std::string& base_dir, const std::string& asset_name) {
        const std::filesystem::path asset_path(asset_name);
        if (asset_path.is_absolute()) {
            return asset_path.string();
        }
        return (std::filesystem::path(base_dir) / asset_path).lexically_normal().string();
    };

    for (size_t i = 0; i < policy_count; i++) {
        const std::string& policy_model_name = model_names[i];
        const std::string policy_motion_name = motion_names.empty() ? "" : motion_names[i];
        const std::string policy_latent_name = latent_names.empty() ? "" : latent_names[i];
        const long int policy_frame_stack = frame_stacks[i];
        if (policy_model_name.empty()) {
            throw std::runtime_error("model_names[" + std::to_string(i) + "] must not be empty");
        }
        if (policy_frame_stack <= 0 ||
            policy_frame_stack > static_cast<long int>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "frame_stacks[" + std::to_string(i) + "] must be a positive 32-bit integer");
        }
        PolicyRuntime policy;
        if (policy_ids.empty()) {
            policy.id = i == 0 ? "locomotion" : std::filesystem::path(policy_model_name).stem().string();
        } else {
            policy.id = policy_ids[i];
        }
        if (policy.id.empty()) {
            throw std::runtime_error("policy_ids[" + std::to_string(i) + "] must not be empty");
        }
        const auto duplicate_id = std::find_if(
            policies_.begin(), policies_.end(), [&policy](const PolicyRuntime& existing) {
                return existing.id == policy.id;
            });
        if (duplicate_id != policies_.end()) {
            throw std::runtime_error("policy_ids must be unique: " + policy.id);
        }
        policy.name = policy_model_name;
        policy.model_path = resolve_asset_path(model_dir, policy_model_name);
        policy.frame_stack = static_cast<int>(policy_frame_stack);
        policy.stack_order = parse_obs_stack_order(obs_stack_orders[i]);
        if (!policy_motion_name.empty()) {
            policy.motion_path = resolve_asset_path(motion_dir, policy_motion_name);
        }
        if (!policy_latent_name.empty()) {
            policy.latent_loader = std::make_unique<LatentLoader>(
                resolve_asset_path(latent_dir, policy_latent_name),
                static_cast<float>(latent_gamma), latent_window_size);
        }
        policy.obs_layout = parse_obs_layout(obs_layouts[i], "obs_layouts[" + std::to_string(i) + "]");
        policy.obs_layout_sizes.reserve(policy.obs_layout.size());
        bool has_sparse_history = false;
        for (const ObsSourceSpec& source : policy.obs_layout) {
            if (source.size > std::numeric_limits<int>::max() - policy.obs_num) {
                throw std::runtime_error("obs_layouts[" + std::to_string(i) + "] is too large");
            }
            policy.obs_layout_sizes.push_back(source.size);
            policy.obs_num += source.size;
            if (source.name == "perception") {
                perception_obs_num_ = source.size;
            }
            if (!source.history_taps.empty()) {
                has_sparse_history = true;
                const auto invalid_tap = std::find_if(
                    source.history_taps.begin(), source.history_taps.end(),
                    [&policy](int tap) { return tap >= policy.frame_stack; });
                if (invalid_tap != source.history_taps.end()) {
                    throw std::runtime_error(
                        "obs_layouts[" + std::to_string(i) + "] history tap " +
                        std::to_string(*invalid_tap) + " for source '" +
                        source.name + "' must be smaller than frame_stacks[" +
                        std::to_string(i) + "]");
                }
                if (policy.stack_order == ObsStackOrder::FrameMajor &&
                    std::adjacent_find(
                        source.history_taps.begin(), source.history_taps.end(),
                        [](int previous, int next) { return previous <= next; }) !=
                        source.history_taps.end()) {
                    throw std::runtime_error(
                        "obs_layouts[" + std::to_string(i) +
                        "] history taps for source '" + source.name +
                        "' must be strictly descending for frame_major");
                }
            }
        }

        const long long dense_history_num =
            static_cast<long long>(policy.obs_num) * policy.frame_stack;
        if (dense_history_num > static_cast<long long>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "obs_layouts[" + std::to_string(i) + "] history buffer is too large");
        }
        policy.obs_input_num = static_cast<int>(dense_history_num);

        if (has_sparse_history) {
            size_t input_offset = 0;
            const auto append_history_slice =
                [&policy, &input_offset](size_t obs_offset,
                                         const ObsSourceSpec& source,
                                         int tap) {
                    const size_t frame = static_cast<size_t>(policy.frame_stack - 1 - tap);
                    policy.history_gather_plan.push_back({
                        frame * static_cast<size_t>(policy.obs_num) + obs_offset,
                        static_cast<size_t>(source.size),
                    });
                    input_offset += static_cast<size_t>(source.size);
                };

            if (policy.stack_order == ObsStackOrder::ObsMajor) {
                size_t obs_offset = 0;
                for (const ObsSourceSpec& source : policy.obs_layout) {
                    if (source.history_taps.empty()) {
                        for (int tap = policy.frame_stack - 1; tap >= 0; tap--) {
                            append_history_slice(obs_offset, source, tap);
                        }
                    } else {
                        for (const int tap : source.history_taps) {
                            append_history_slice(obs_offset, source, tap);
                        }
                    }
                    obs_offset += static_cast<size_t>(source.size);
                }
            } else {
                for (int tap = policy.frame_stack - 1; tap >= 0; tap--) {
                    size_t obs_offset = 0;
                    for (const ObsSourceSpec& source : policy.obs_layout) {
                        if (source.history_taps.empty() ||
                            std::find(source.history_taps.begin(),
                                      source.history_taps.end(), tap) !=
                                source.history_taps.end()) {
                            append_history_slice(obs_offset, source, tap);
                        }
                        obs_offset += static_cast<size_t>(source.size);
                    }
                }
            }
            policy.obs_input_num = static_cast<int>(input_offset);
        }
        if (!policy.motion_path.empty()) {
            motion_policy_indices_.push_back(static_cast<int>(policies_.size()));
        }
        policies_.push_back(std::move(policy));
    }

    locomotion_policy_idx_ = find_policy_by_id("locomotion");
    if (locomotion_policy_idx_ < 0) {
        throw std::runtime_error("policy_ids must include exactly one locomotion policy");
    }
    if (!policies_[locomotion_policy_idx_].motion_path.empty()) {
        throw std::runtime_error("The locomotion policy must not have a motion asset");
    }
    active_policy_idx_ = locomotion_policy_idx_;

    RCLCPP_INFO(this->get_logger(), "robot_name: %s", robot_name.c_str());
    RCLCPP_INFO(this->get_logger(), "policy_name: %s", policy_name.c_str());
    RCLCPP_INFO(this->get_logger(), "robot_config: %s", robot_config_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "model_dir: %s", model_dir.c_str());
    RCLCPP_INFO(this->get_logger(), "motion_dir: %s", motion_dir.c_str());
    RCLCPP_INFO(this->get_logger(), "latent_dir: %s", latent_dir.c_str());
    for(size_t i = 0; i < policies_.size(); i++) {
        RCLCPP_INFO(this->get_logger(), "policy_id %zu: %s", i, policies_[i].id.c_str());
        RCLCPP_INFO(this->get_logger(), "policy %zu: %s", i, policies_[i].name.c_str());
        RCLCPP_INFO(this->get_logger(), "policy_model_path %zu: %s", i, policies_[i].model_path.c_str());
        if (!policies_[i].motion_path.empty()) {
            RCLCPP_INFO(this->get_logger(), "policy_motion_path %zu: %s", i, policies_[i].motion_path.c_str());
        }
    }
    RCLCPP_INFO(this->get_logger(), "act_alpha: %f", act_alpha_);
    RCLCPP_INFO(this->get_logger(), "intra_threads: %d", intra_threads_);
    RCLCPP_INFO(this->get_logger(), "supports_interrupt: %s", has_obs_source("interrupt") ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "has_motion_policy: %s", motion_policy_indices_.empty() ? "false" : "true");
    RCLCPP_INFO(this->get_logger(), "perception_obs_num: %d", perception_obs_num_);
    RCLCPP_INFO(this->get_logger(), "perception_obs_topic: %s", perception_obs_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "use_depth: %s", use_depth_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "joint_num: %d", joint_num_);
    RCLCPP_INFO(this->get_logger(), "decimation: %d", decimation_);
    RCLCPP_INFO(this->get_logger(), "dt: %f", dt_);
    RCLCPP_INFO(this->get_logger(), "cmd_vel_timeout_s: %f", cmd_vel_timeout_s_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_lin_vel: %f", obs_scales_lin_vel_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_ang_vel: %f", obs_scales_ang_vel_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_dof_pos: %f", obs_scales_dof_pos_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_dof_vel: %f", obs_scales_dof_vel_);
    RCLCPP_INFO(this->get_logger(), "obs_scales_gravity_b: %f", obs_scales_gravity_b_);
    print_vector<double>("action_scale", action_scale_);
    RCLCPP_INFO(this->get_logger(), "clip_actions: %f", clip_actions_);
    print_vector<long int>("usd2urdf", usd2urdf_);
    print_vector<double>("clip_cmd", clip_cmd_);
    print_vector<double>("joint_default_angle", joint_default_angle_);
    print_vector<double>("joint_limits", joint_limits_);
    RCLCPP_INFO(this->get_logger(), "gravity_z_upper: %f", gravity_z_upper_);
}

void InferenceNode::subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg) {
    if (msg->axes.size() < 6 || msg->buttons.size() < 6) {
        zero_cmd_vel();
        set_runtime_fault("malformed_joy_input");
        RCLCPP_WARN(this->get_logger(), "Ignoring malformed Joy message");
        return;
    }
    if (!std::all_of(msg->axes.begin(), msg->axes.begin() + 6,
                     [](float value) { return std::isfinite(value); })) {
        zero_cmd_vel();
        set_runtime_fault("nonfinite_joy_input");
        RCLCPP_WARN(this->get_logger(), "Rejected non-finite Joy message");
        return;
    }
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (is_joy_control_){
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        cmd_vel_[0] = std::clamp(msg->axes[4] * clip_cmd_[1], clip_cmd_[0], clip_cmd_[1]);
        cmd_vel_[1] = std::clamp(msg->axes[3] * clip_cmd_[3], clip_cmd_[2], clip_cmd_[3]);
            if (msg->axes[2] < 0) {
            cmd_vel_[2] = std::clamp(-msg->axes[2] * clip_cmd_[5], clip_cmd_[4], clip_cmd_[5]);
            } else if (msg->axes[5] < 0) {
            cmd_vel_[2] = std::clamp(msg->axes[5] * clip_cmd_[5], clip_cmd_[4], clip_cmd_[5]);
            } else {
            cmd_vel_[2] = 0.0;
        }
    }
    if ((msg->buttons[2] == 1 && msg->buttons[2] != last_button0_)) {
        request_motion_cancel();
        zero_cmd_vel();
        if (is_running_.load()){
            reset_runtime_state();
            RCLCPP_INFO(this->get_logger(), "Inference paused");
        }
        try {
            if (robot_->is_init_.load()){
                robot_->deinit_motors();
                RCLCPP_INFO(this->get_logger(), "Motors deinitialized");
            } else {
                robot_->init_motors();
                RCLCPP_INFO(this->get_logger(), "Motors initialized");
            }
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "Failed to change motor state: %s", e.what());
        }
    }
    if (msg->buttons[0] == 1 && msg->buttons[0] != last_button1_) {
        request_motion_cancel();
        zero_cmd_vel();
        if (is_running_.load()){
            reset_runtime_state();
            RCLCPP_INFO(this->get_logger(), "Inference paused");
        }
        try {
            if (!start_joint_reset()) {
                RCLCPP_WARN(this->get_logger(), "Motor reset is already in progress");
            }
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "Failed to start motor reset: %s", e.what());
        }
    }
    if (msg->buttons[1] == 1 && msg->buttons[1] != last_button2_) {
        if (is_running_.load()) {
            request_motion_cancel();
            zero_cmd_vel();
            is_running_.store(false);
            RCLCPP_INFO(this->get_logger(), "Inference paused");
        } else if (!robot_->is_init_.load()) {
            RCLCPP_WARN(this->get_logger(), "Motors are not initialized, cannot start inference");
        } else {
            std::string blocked_reason;
            if (!try_start_inference(blocked_reason)) {
                RCLCPP_WARN(this->get_logger(), "Cannot start inference: %s", blocked_reason.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), "Inference started");
            }
        }
    }
    if (msg->buttons[3] == 1 && msg->buttons[3] != last_button3_) {
        is_joy_control_.store(!is_joy_control_);
        zero_cmd_vel();
        if (motion_action_active_.load()) {
            request_motion_cancel();
        }
        RCLCPP_INFO(this->get_logger(), "Controlled by %s", is_joy_control_.load() ? "joy" : "/cmd_vel");
    }
    if (supports_interrupt() || has_motion_policy()) {
        if (msg->buttons[4] == 1 && msg->buttons[4] != last_button4_) {
            if (motion_action_active_.load()) {
                request_motion_cancel();
                RCLCPP_WARN(this->get_logger(), "Joystick override requested motion cancellation");
            } else {
            const auto switch_while_paused = [this](auto&& switch_mode) {
                std::unique_lock<std::mutex> switch_lock(lb_switch_mutex_);
                const bool restore_running = is_running_.exchange(false);
                if (restore_running) {
                    RCLCPP_INFO(this->get_logger(), "Inference paused");
                }
                try {
                    switch_mode();
                } catch (...) {
                    if (restore_running) {
                        std::string blocked_reason;
                        if (resume_inference_if_fault_free(blocked_reason)) {
                            RCLCPP_INFO(this->get_logger(), "Inference started");
                        } else {
                            RCLCPP_WARN(this->get_logger(), "Inference remains stopped: %s", blocked_reason.c_str());
                        }
                    }
                    throw;
                }
                if (restore_running) {
                    std::string blocked_reason;
                    if (resume_inference_if_fault_free(blocked_reason)) {
                        RCLCPP_INFO(this->get_logger(), "Inference started");
                    } else {
                        RCLCPP_WARN(this->get_logger(), "Inference remains stopped: %s", blocked_reason.c_str());
                    }
                }
            };
            if (supports_interrupt()) {
                switch_while_paused([this]() {
                    std::unique_lock<std::mutex> lock(mode_mutex_);
                    is_interrupt_.store(!is_interrupt_.load());
                    RCLCPP_INFO(this->get_logger(), "Interrupt mode %s", is_interrupt_.load() ? "enabled" : "disabled");
                });
            } else if (has_motion_policy()) {
                switch_while_paused([this]() {
                    std::string policy_name;
                    std::unique_lock<std::mutex> lock(mode_mutex_);
                    is_motion_policy_.store(!is_motion_policy_.load());
                    active_policy_idx_ = is_motion_policy_.load()
                        ? motion_policy_indices_[current_motion_policy_idx_]
                        : locomotion_policy_idx_;
                    reset_policy_runtime(active_policy());
                    policy_name = active_policy().name;
                    RCLCPP_INFO(this->get_logger(), "Policy enabled: %s", policy_name.c_str());
                });
            }
            }
        }
        last_button4_ = msg->buttons[4];
    }
    if (has_motion_policy()) {
        if (msg->buttons[5] == 1 && msg->buttons[5] != last_button5_) {
            if (motion_action_active_.load()) {
                request_motion_cancel();
                RCLCPP_WARN(this->get_logger(), "Joystick override requested motion cancellation");
            } else {
            std::unique_lock<std::mutex> lock(mode_mutex_);
            if (is_motion_policy_.load()) {
                RCLCPP_WARN(this->get_logger(), "Cannot switch motion policy while in motion policy mode");
            } else {
                current_motion_policy_idx_ = (current_motion_policy_idx_ + 1) % motion_policy_indices_.size();
                RCLCPP_INFO(this->get_logger(), "Selected policy: %s", policies_[motion_policy_indices_[current_motion_policy_idx_]].name.c_str());
            }
            }
        }
        last_button5_ = msg->buttons[5];
    }
    last_button0_ = msg->buttons[2];
    last_button1_ = msg->buttons[0];
    last_button2_ = msg->buttons[1];
    last_button3_ = msg->buttons[3];
}

void InferenceNode::subs_cmd_callback(const std::shared_ptr<geometry_msgs::msg::Twist> msg){
    // Serialize the source check and write with source changes. Without this
    // lease lock, an external callback could pass the check, lose ownership,
    // then write a stale command after the switch to joystick control.
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if(!is_joy_control_){
        if (!std::isfinite(msg->linear.x) || !std::isfinite(msg->linear.y) ||
            !std::isfinite(msg->angular.z)) {
            zero_cmd_vel();
            set_runtime_fault("nonfinite_cmd_vel_input");
            RCLCPP_WARN(this->get_logger(), "Rejected non-finite /cmd_vel");
            return;
        }
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        cmd_vel_[0] = std::clamp(msg->linear.x, clip_cmd_[0], clip_cmd_[1]);
        cmd_vel_[1] = std::clamp(msg->linear.y, clip_cmd_[2], clip_cmd_[3]);
        cmd_vel_[2] = std::clamp(msg->angular.z, clip_cmd_[4], clip_cmd_[5]);
        last_external_cmd_time_ = std::chrono::steady_clock::now();
        external_cmd_seen_ = true;
    }
}

void InferenceNode::subs_perception_callback(const std::shared_ptr<std_msgs::msg::Float32MultiArray> msg){
    if(perception_obs_num_ > 0){
        if (msg->data.size() < perception_obs_buffer_.size()) {
            RCLCPP_WARN(this->get_logger(), "Perception obs message too small: got %zu, expected %zu", msg->data.size(), perception_obs_buffer_.size());
            {
                std::unique_lock<std::mutex> lock(perception_mutex_);
                std::fill(perception_obs_buffer_.begin(), perception_obs_buffer_.end(), 0.0f);
            }
            set_runtime_fault("malformed_perception_input");
            return;
        }
        if (!std::all_of(
                msg->data.begin(), msg->data.begin() + perception_obs_buffer_.size(),
                [](float value) { return std::isfinite(value); })) {
            {
                std::unique_lock<std::mutex> lock(perception_mutex_);
                std::fill(perception_obs_buffer_.begin(), perception_obs_buffer_.end(), 0.0f);
            }
            set_runtime_fault("nonfinite_perception_input");
            RCLCPP_WARN(this->get_logger(), "Rejected non-finite perception observation");
            return;
        }
        std::unique_lock<std::mutex> lock(perception_mutex_);
        std::copy(msg->data.begin(), msg->data.begin() + perception_obs_buffer_.size(), perception_obs_buffer_.begin());
    }
}

void InferenceNode::subs_joint_state_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg){
    if(supports_interrupt() && is_interrupt_.load()){
        if (msg->position.size() < interrupt_action_.size()) {
            set_runtime_fault("malformed_interrupt_joint_state");
            RCLCPP_WARN(this->get_logger(), "Interrupt joint state is too small");
            return;
        }
        if (!std::all_of(
                msg->position.begin(), msg->position.begin() + interrupt_action_.size(),
                [](double value) { return std::isfinite(value); })) {
            set_runtime_fault("nonfinite_interrupt_joint_state");
            RCLCPP_WARN(this->get_logger(), "Rejected non-finite interrupt joint state");
            return;
        }
        std::unique_lock<std::mutex> lock(interrupt_mutex_);
        for(size_t i = 0; i < interrupt_action_.size(); i++){
            interrupt_action_[i] = msg->position[i];
        }
    }
}

bool InferenceNode::start_joint_reset() {
    std::unique_lock<std::mutex> lock(reset_thread_mutex_);
    if (reset_thread_running_) {
        return false;
    }
    if (!robot_->is_init_.load()) {
        throw std::runtime_error("Motors are not initialized");
    }
    if (reset_thread_.joinable()) {
        reset_thread_.join();
    }

    reset_thread_running_ = true;
    try {
        reset_thread_ = std::thread([
            this, robot = robot_, joint_default_angle = joint_default_angle_, logger = this->get_logger()
        ]() {
            try {
                robot->reset_joints(joint_default_angle);
                if (robot->is_init_.load()) {
                    RCLCPP_INFO(logger, "Motors reset");
                } else {
                    RCLCPP_INFO(logger, "Motor reset interrupted by deinitialization");
                }
            } catch (const std::exception& e) {
                if (robot->is_init_.load()) {
                    RCLCPP_WARN(logger, "Failed to reset motors: %s", e.what());
                } else {
                    RCLCPP_INFO(logger, "Motor reset interrupted by deinitialization");
                }
            } catch (...) {
                RCLCPP_ERROR(logger, "Motor reset failed with an unknown exception");
            }

            std::lock_guard<std::mutex> state_lock(reset_thread_mutex_);
            reset_thread_running_ = false;
        });
    } catch (...) {
        reset_thread_running_ = false;
        throw;
    }
    return true;
}

void InferenceNode::reset_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                     std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
    try {
        if (is_running_.load()){
            reset_runtime_state();
            RCLCPP_INFO(this->get_logger(), "Inference paused");
        }
        if (!start_joint_reset()) {
            response->success = false;
            response->message = "Joint reset is already in progress";
            return;
        }
        response->success = true;
        response->message = "Joint reset started";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::refresh_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                     std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    try {
        robot_->refresh_joints();
        response->success = true;
        response->message = "Motors refreshed successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::read_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                     std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    try {
        robot_->read_joints();
        response->success = true;
        response->message = "Joints read successfully";
        publish_joint_states();
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::read_imu_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                 std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    try {
        robot_->read_imu();
        response->success = true;
        response->message = "IMU read successfully";
        publish_imu();
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::set_zeros_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                  std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (is_running_.load()) {
        response->success = false;
        response->message = "Inference is running, cannot set zeros";
        return;
    }
    try {
        robot_->set_zeros();
        response->success = true;
        response->message = "Zeros set successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::clear_errors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                     std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    try {
        robot_->clear_errors();
        response->success = true;
        response->message = "Errors cleared successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::init_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
    try {
        robot_->init_motors();
        response->success = true;
        response->message = "Motors initialized successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::deinit_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
    try {
        request_motion_cancel();
        zero_cmd_vel();
        if (is_running_.load()){
            reset_runtime_state();
            RCLCPP_INFO(this->get_logger(), "Inference paused");
        }
        robot_->deinit_motors();
        response->success = true;
        response->message = "Motors deinitialized successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void InferenceNode::start_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (is_running_.load()) {
        response->success = false;
        response->message = "Inference is already running";
        return;
    }
    if (!robot_->is_init_.load()) {
        response->success = false;
        response->message = "Motors are not initialized, cannot start inference";
        RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
        return;
    }
    std::string blocked_reason;
    if (!try_start_inference(blocked_reason)) {
        response->success = false;
        response->message = blocked_reason;
        RCLCPP_WARN(this->get_logger(), "Cannot start inference: %s", blocked_reason.c_str());
        return;
    }
    response->success = true;
    response->message = "Inference started";
}

void InferenceNode::stop_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                       std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
    request_motion_cancel();
    zero_cmd_vel();
    const bool was_running = is_running_.exchange(false);
    response->success = true;
    response->message = was_running ? "Inference stopped" : "Inference was already stopped";
}

void InferenceNode::set_command_source_srv(
    const std::shared_ptr<roboparty_inference::srv::SetCommandSource::Request> request,
    std::shared_ptr<roboparty_inference::srv::SetCommandSource::Response> response) {
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
    const auto joystick = roboparty_inference::srv::SetCommandSource::Request::JOYSTICK;
    const auto external = roboparty_inference::srv::SetCommandSource::Request::EXTERNAL;
    if (request->source != joystick && request->source != external) {
        response->success = false;
        response->message = "Unknown command source";
        return;
    }
    const bool use_joystick = request->source == joystick;
    if (is_joy_control_.load() == use_joystick) {
        response->success = true;
        response->message = use_joystick
            ? "Joystick control already selected"
            : "External command control already selected";
        return;
    }
    if (motion_action_active_.load()) {
        request_motion_cancel();
    }
    zero_cmd_vel();
    is_joy_control_.store(use_joystick);
    response->success = true;
    response->message = is_joy_control_.load() ? "Joystick control selected" : "External command control selected";
    RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
}

rclcpp_action::GoalResponse InferenceNode::handle_motion_goal(
    const rclcpp_action::GoalUUID& uuid,
    std::shared_ptr<const ExecuteMotion::Goal> goal) {
    if (!goal || goal->motion_id.empty() || !std::isfinite(goal->timeout_s) ||
        goal->timeout_s < 0.1f || goal->timeout_s > 300.0f) {
        RCLCPP_WARN(this->get_logger(), "Rejected malformed motion goal");
        return rclcpp_action::GoalResponse::REJECT;
    }
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (!robot_->is_init_.load() || !is_running_.load()) {
        RCLCPP_WARN(this->get_logger(), "Rejected motion goal: motors/inference are not ready");
        return rclcpp_action::GoalResponse::REJECT;
    }
    {
        std::unique_lock<std::mutex> reset_lock(reset_thread_mutex_);
        if (reset_thread_running_) {
            RCLCPP_WARN(this->get_logger(), "Rejected motion goal during joint reset");
            return rclcpp_action::GoalResponse::REJECT;
        }
    }
    const int policy_idx = find_policy_by_id(goal->motion_id);
    if (policy_idx < 0 || !policies_[policy_idx].motion_loader) {
        RCLCPP_WARN(this->get_logger(), "Rejected unknown motion id: %s", goal->motion_id.c_str());
        return rclcpp_action::GoalResponse::REJECT;
    }
    std::unique_lock<std::mutex> goal_lock(motion_goal_mutex_);
    bool expected = false;
    if (!motion_action_active_.compare_exchange_strong(expected, true)) {
        RCLCPP_WARN(this->get_logger(), "Rejected motion goal: another motion is active");
        return rclcpp_action::GoalResponse::REJECT;
    }
    active_motion_goal_uuid_ = uuid;
    motion_cancel_requested_.store(false);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse InferenceNode::handle_motion_cancel(
    const std::shared_ptr<ExecuteMotionGoalHandle> goal_handle) {
    std::unique_lock<std::mutex> goal_lock(motion_goal_mutex_);
    if (!motion_action_active_.load() ||
        goal_handle->get_goal_id() != active_motion_goal_uuid_) {
        return rclcpp_action::CancelResponse::REJECT;
    }
    motion_cancel_requested_.store(true);
    return rclcpp_action::CancelResponse::ACCEPT;
}

void InferenceNode::handle_motion_accepted(
    const std::shared_ptr<ExecuteMotionGoalHandle> goal_handle) {
    if (motion_action_thread_.joinable()) {
        motion_action_thread_.join();
    }
    motion_action_thread_ = std::thread(&InferenceNode::execute_motion, this, goal_handle);
}

void InferenceNode::execute_motion(
    const std::shared_ptr<ExecuteMotionGoalHandle> goal_handle) {
    try {
        const auto goal = goal_handle->get_goal();
        std::string error;
        if (!activate_motion_policy(goal->motion_id, goal_handle->get_goal_id(), error)) {
            finish_motion(goal_handle, false, error, "motion activation failed");
            return;
        }

        const auto started_at = std::chrono::steady_clock::now();
        const auto poll_period = std::chrono::milliseconds(20);
        while (rclcpp::ok()) {
            if (goal_handle->is_canceling() || motion_cancel_requested_.load()) {
                finish_motion(
                    goal_handle,
                    false,
                    "Motion cancelled by client or runtime override",
                    "motion cancelled");
                return;
            }
            if (!robot_->is_init_.load() || !is_running_.load()) {
                finish_motion(
                    goal_handle,
                    false,
                    "Motors or inference stopped during motion",
                    "runtime no longer ready");
                return;
            }

            size_t frame = 0;
            size_t frame_count = 0;
            bool complete = false;
            {
                std::unique_lock<std::mutex> mode_lock(mode_mutex_);
                const auto& policy = active_policy();
                if (policy.id != goal->motion_id || !policy.motion_loader) {
                    error = "Active policy changed unexpectedly";
                } else {
                    frame = policy.motion_frames_executed;
                    frame_count = policy.motion_loader->get_num_frames();
                    complete = policy.motion_complete &&
                        applied_action_generation_.load(std::memory_order_acquire) >=
                            policy.motion_final_action_generation;
                }
            }
            if (!error.empty()) {
                finish_motion(goal_handle, false, error, "motion ownership lost");
                return;
            }

            auto feedback = std::make_shared<ExecuteMotion::Feedback>();
            feedback->motion_id = goal->motion_id;
            feedback->frames_executed = static_cast<uint32_t>(std::min(frame, frame_count));
            feedback->total_frames = static_cast<uint32_t>(frame_count);
            feedback->progress = frame_count == 0 ? 0.0f :
                static_cast<float>(std::min(frame, frame_count)) / static_cast<float>(frame_count);
            goal_handle->publish_feedback(feedback);

            if (complete) {
                finish_motion(goal_handle, true, "Motion completed", "motion completed");
                return;
            }
            const double elapsed_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started_at).count();
            if (elapsed_s > goal->timeout_s) {
                finish_motion(goal_handle, false, "Motion timed out", "motion timeout");
                return;
            }
            std::this_thread::sleep_for(poll_period);
        }
        finish_motion(goal_handle, false, "ROS shutdown", "ROS shutdown");
    } catch (const std::exception& error) {
        RCLCPP_ERROR(this->get_logger(), "Motion action failed: %s", error.what());
        finish_motion(goal_handle, false, error.what(), "motion exception");
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Motion action failed with an unknown exception");
        finish_motion(goal_handle, false, "Unknown motion error", "motion exception");
    }
}

void InferenceNode::finish_motion(
    const std::shared_ptr<ExecuteMotionGoalHandle> goal_handle,
    bool requested_success,
    const std::string& message,
    const std::string& return_reason) noexcept {
    try {
        auto result = std::make_shared<ExecuteMotion::Result>();
        result->success = requested_success;
        result->message = message;
        try {
            return_to_locomotion(return_reason);
        } catch (const std::exception& error) {
            requested_success = false;
            result->success = false;
            result->message += std::string("; failed to return to locomotion: ") + error.what();
            RCLCPP_ERROR(this->get_logger(), "%s", result->message.c_str());
        } catch (...) {
            requested_success = false;
            result->success = false;
            result->message += "; failed to return to locomotion";
            RCLCPP_ERROR(this->get_logger(), "%s", result->message.c_str());
        }

        std::unique_lock<std::mutex> goal_lock(motion_goal_mutex_);
        if (!motion_action_active_.load() ||
            goal_handle->get_goal_id() != active_motion_goal_uuid_) {
            RCLCPP_ERROR(this->get_logger(), "Refusing to finalize a stale motion goal");
            return;
        }
        const bool client_cancelled = goal_handle->is_canceling();
        const bool runtime_cancelled = motion_cancel_requested_.load();
        const bool runtime_ready = robot_->is_init_.load() && is_running_.load();
        const bool can_succeed =
            requested_success && !client_cancelled && !runtime_cancelled && runtime_ready;
        result->success = can_succeed;
        try {
            if (client_cancelled) {
                result->message = "Motion cancelled by client";
                goal_handle->canceled(result);
            } else if (can_succeed) {
                goal_handle->succeed(result);
            } else {
                goal_handle->abort(result);
            }
        } catch (const std::exception& error) {
            RCLCPP_ERROR(this->get_logger(), "Failed to finalize motion action: %s", error.what());
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "Failed to finalize motion action");
        }
        motion_action_active_.store(false);
        active_motion_goal_uuid_.fill(0U);
    } catch (const std::exception& error) {
        RCLCPP_ERROR(this->get_logger(), "Unhandled motion finalization error: %s", error.what());
        motion_action_active_.store(false);
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unhandled motion finalization error");
        motion_action_active_.store(false);
    }
}

void InferenceNode::publish_runtime_state() {
    roboparty_inference::msg::RuntimeState state;
    state.motors_initialized = robot_->is_init_.load();
    state.inference_running = is_running_.load();
    state.command_source = is_joy_control_.load()
        ? roboparty_inference::msg::RuntimeState::COMMAND_SOURCE_JOYSTICK
        : roboparty_inference::msg::RuntimeState::COMMAND_SOURCE_EXTERNAL;
    state.motion_active = motion_action_active_.load() || is_motion_policy_.load();
    state.external_command_fresh = external_command_is_fresh();
    {
        std::unique_lock<std::mutex> fault_lock(runtime_fault_mutex_);
        state.fault = runtime_fault_;
    }
    {
        std::unique_lock<std::mutex> mode_lock(mode_mutex_);
        const auto& policy = active_policy();
        state.active_policy_id = policy.id;
        state.motion_frames_executed = static_cast<uint32_t>(policy.motion_frames_executed);
        state.motion_frame_count = policy.motion_loader
            ? static_cast<uint32_t>(policy.motion_loader->get_num_frames())
            : 0U;
    }
    runtime_state_publisher_->publish(state);
}

void InferenceNode::publish_terminal_fault_state(const std::string& fault) {
    // This path is called from worker exception handlers and must not acquire
    // mode_mutex_: a worker may already hold it.  The deliberately minimal
    // terminal snapshot is published synchronously before shutdown can stop
    // the 100 ms periodic timer.
    roboparty_inference::msg::RuntimeState state;
    state.motors_initialized = robot_ && robot_->is_init_.load();
    state.inference_running = false;
    state.command_source = is_joy_control_.load()
        ? roboparty_inference::msg::RuntimeState::COMMAND_SOURCE_JOYSTICK
        : roboparty_inference::msg::RuntimeState::COMMAND_SOURCE_EXTERNAL;
    state.active_policy_id = "";
    state.motion_active = motion_action_active_.load() || is_motion_policy_.load();
    state.motion_frames_executed = 0U;
    state.motion_frame_count = 0U;
    state.external_command_fresh = false;
    state.fault = fault;
    if (runtime_state_publisher_) {
        runtime_state_publisher_->publish(state);
    }
}

void InferenceNode::publish_joint_states() {
    joint_pos_buffer_ = robot_->get_joint_q();
    joint_vel_buffer_ = robot_->get_joint_vel();
    joint_torques_buffer_ = robot_->get_joint_tau();
    joint_state_msg_.header.stamp = this->now();
    joint_state_msg_.effort.resize(joint_num_);
    for (int i = 0; i < joint_num_; i++) {
        joint_state_msg_.position[i] = joint_pos_buffer_[i];
        joint_state_msg_.velocity[i] = joint_vel_buffer_[i];
        joint_state_msg_.effort[i] = joint_torques_buffer_[i];
    }
    joint_state_publisher_->publish(joint_state_msg_);
}

void InferenceNode::publish_action() {
    action_msg_.header.stamp = this->now();
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        for (int i = 0; i < joint_num_; i++) {
            action_msg_.position[i] = act_[i];
        }
    }
    action_publisher_->publish(action_msg_);
}

void InferenceNode::publish_imu() {
    const auto quat = robot_->get_quat();
    const auto ang_vel = robot_->get_ang_vel();
    auto msg = sensor_msgs::msg::Imu();
    msg.header.stamp = this->now();
    msg.orientation.w = quat[0];
    msg.orientation.x = quat[1];
    msg.orientation.y = quat[2];
    msg.orientation.z = quat[3];
    msg.angular_velocity.x = ang_vel[0];
    msg.angular_velocity.y = ang_vel[1];
    msg.angular_velocity.z = ang_vel[2];
    imu_publisher_->publish(msg);
}
