// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <sys/mman.h>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <Eigen/Geometry>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <queue>
#include <sstream>
#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp> 
#include "utils/motion_loader.hpp"
#include "utils/latent_loader.hpp"
#include <std_srvs/srv/trigger.hpp>
#include "roboparty_inference/action/execute_motion.hpp"
#include "roboparty_inference/msg/runtime_state.hpp"
#include "roboparty_inference/srv/set_command_source.hpp"
#include "robot_interface.hpp"

enum class ObsStackOrder {
    FrameMajor,
    ObsMajor,
};

class InferenceNode;

struct ObsSourceDefinition {
    const char* name;
    void (InferenceNode::*get)(std::vector<float>& segment);
};

struct ObsSourceSpec {
    std::string name;
    const ObsSourceDefinition* source;
    int size;
    // Empty keeps legacy contiguous stacking; otherwise values are explicit inference-tick lags.
    std::vector<int> history_taps;
};

struct ObsHistorySlice {
    size_t history_offset;
    size_t size;
};

class InferenceNode : public rclcpp::Node {
   public:
    struct ModelContext {
        std::unique_ptr<Ort::Session> session;
        std::unique_ptr<Ort::MemoryInfo> memory_info;
        std::unique_ptr<Ort::Value> input_tensor;
        std::unique_ptr<Ort::Value> output_tensor;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<const char *> input_names_raw;
        std::vector<const char *> output_names_raw;
        std::vector<int64_t> input_shape;
        std::vector<int64_t> output_shape;
        std::vector<float> input_buffer;
        std::vector<float> output_buffer;
        size_t num_inputs;
        size_t num_outputs;
    };

    struct PolicyRuntime {
        std::string id;
        std::string name;
        std::string model_path;
        std::string motion_path;
        std::vector<ObsSourceSpec> obs_layout;
        std::vector<int> obs_layout_sizes;
        std::vector<std::vector<float>> obs_segments;
        std::vector<float> obs;
        // Sparse mode only: complete frames ordered from oldest to newest.
        std::vector<float> obs_history;
        // Non-empty selects sparse mode and describes sequential copies into the model input.
        std::vector<ObsHistorySlice> history_gather_plan;
        int obs_num = 0;
        int obs_input_num = 0;
        int frame_stack = 1;
        ObsStackOrder stack_order = ObsStackOrder::FrameMajor;
        std::unique_ptr<ModelContext> ctx;
        std::shared_ptr<MotionLoader> motion_loader;
        std::unique_ptr<LatentLoader> latent_loader;
        size_t motion_frame = 0;
        size_t motion_frames_executed = 0;
        uint64_t motion_final_action_generation = 0;
        bool motion_complete = false;
        bool is_first_frame = true;
    };

    InferenceNode() : Node("inference_node") {
        load_config();

        robot_ = std::make_shared<RobotInterface>(robot_config_path_);

        Ort::ThreadingOptions thread_opts;
        if (intra_threads_ > 0) {
            thread_opts.SetGlobalIntraOpNumThreads(intra_threads_);
        }
        env_ = std::make_unique<Ort::Env>(thread_opts, ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeInference");
        if (policies_.empty()) {
            throw std::runtime_error("At least one policy must be configured");
        }
        for (size_t i = 0; i < policies_.size(); i++) {
            PolicyRuntime& policy = policies_[i];
            policy.obs.resize(policy.obs_num, 0.0f);
            policy.obs_segments.resize(policy.obs_layout.size());
            for (size_t j = 0; j < policy.obs_layout.size(); j++) {
                policy.obs_segments[j].resize(policy.obs_layout[j].size, 0.0f);
            }
            if (!policy.history_gather_plan.empty()) {
                policy.obs_history.resize(
                    static_cast<size_t>(policy.obs_num) * static_cast<size_t>(policy.frame_stack),
                    0.0f);
            }
            if (!policy.motion_path.empty()) {
                policy.motion_loader = std::make_shared<MotionLoader>(policy.motion_path);
                if (policy.motion_loader->get_num_frames() == 0) {
                    throw std::runtime_error("Motion file has no frames: " + policy.motion_path);
                }
                if (policy.motion_loader->get_num_joints() != static_cast<size_t>(joint_num_)) {
                    throw std::runtime_error("Motion joint count mismatch: " + policy.motion_path);
                }
            }
            setup_model(policy.ctx, policy.model_path, policy.obs_input_num);
        }
        initialize_runtime_state();
        reset_runtime_state();

        auto data_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", data_qos, std::bind(&InferenceNode::subs_joy_callback, this, std::placeholders::_1));
        cmd_subscription_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", data_qos, std::bind(&InferenceNode::subs_cmd_callback,this, std::placeholders::_1
        ));
        if (has_obs_source("perception")) {
            perception_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
                perception_obs_topic_, data_qos,
                std::bind(&InferenceNode::subs_perception_callback, this, std::placeholders::_1));
        }
        if (use_depth_) {
            clear_depth_history_client_ =
                this->create_client<std_srvs::srv::Trigger>("clear_depth_history");
        }
        joint_state_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_ref_states", data_qos,
            std::bind(&InferenceNode::subs_joint_state_callback, this, std::placeholders::_1));
        action_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/action", data_qos);
        imu_publisher_ =
            this->create_publisher<sensor_msgs::msg::Imu>("/imu", data_qos);
        joint_state_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", data_qos);
        reset_joints_service_ = this->create_service<std_srvs::srv::Trigger>(
            "reset_joints", std::bind(&InferenceNode::reset_joints_srv, this, std::placeholders::_1, std::placeholders::_2));
        set_zeros_service_ = this->create_service<std_srvs::srv::Trigger>(
            "set_zeros", std::bind(&InferenceNode::set_zeros_srv, this, std::placeholders::_1, std::placeholders::_2));
        clear_errors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "clear_errors", std::bind(&InferenceNode::clear_errors_srv, this, std::placeholders::_1, std::placeholders::_2));
        refresh_joints_service_ = this->create_service<std_srvs::srv::Trigger>(
            "refresh_joints", std::bind(&InferenceNode::refresh_joints_srv, this, std::placeholders::_1, std::placeholders::_2));
        read_joints_service_ = this->create_service<std_srvs::srv::Trigger>(
            "read_joints", std::bind(&InferenceNode::read_joints_srv, this, std::placeholders::_1, std::placeholders::_2));
        read_imu_service_ = this->create_service<std_srvs::srv::Trigger>(
            "read_imu", std::bind(&InferenceNode::read_imu_srv, this, std::placeholders::_1, std::placeholders::_2));
        init_motors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "init_motors", std::bind(&InferenceNode::init_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        deinit_motors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "deinit_motors", std::bind(&InferenceNode::deinit_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        start_inference_service_ = this->create_service<std_srvs::srv::Trigger>(
            "start_inference", std::bind(&InferenceNode::start_inference_srv, this, std::placeholders::_1, std::placeholders::_2));
        stop_inference_service_ = this->create_service<std_srvs::srv::Trigger>(
            "stop_inference", std::bind(&InferenceNode::stop_inference_srv, this, std::placeholders::_1, std::placeholders::_2));
        set_command_source_service_ = this->create_service<roboparty_inference::srv::SetCommandSource>(
            "set_command_source", std::bind(&InferenceNode::set_command_source_srv, this, std::placeholders::_1, std::placeholders::_2));
        runtime_state_publisher_ = this->create_publisher<roboparty_inference::msg::RuntimeState>(
            "runtime_state", rclcpp::QoS(1).reliable().transient_local());
        runtime_state_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&InferenceNode::publish_runtime_state, this));
        execute_motion_action_server_ = rclcpp_action::create_server<roboparty_inference::action::ExecuteMotion>(
            this,
            "execute_motion",
            std::bind(&InferenceNode::handle_motion_goal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&InferenceNode::handle_motion_cancel, this, std::placeholders::_1),
            std::bind(&InferenceNode::handle_motion_accepted, this, std::placeholders::_1));
        // Worker faults must never race ahead of the transient-local status
        // publisher.  Start real-time work only after every control endpoint is
        // ready, so a terminal startup fault is observable immediately.
        inference_thread_ = std::thread(&InferenceNode::inference, this);
        control_thread_ = std::thread(&InferenceNode::control, this);
    }
    ~InferenceNode() {
        motion_cancel_requested_.store(true);
        is_running_.store(false);
        if (reset_thread_.joinable()) {
            reset_thread_.join();
        }
        if (inference_thread_.joinable()) {
            inference_thread_.join();
        }
        if (control_thread_.joinable()) {
            control_thread_.join();
        }
        if (motion_action_thread_.joinable()) {
            motion_action_thread_.join();
        }
        reset_runtime_state();
        if (robot_) {
            robot_.reset();
        }
    }
    bool supports_interrupt() const;
    bool has_motion_policy() const;
   private:
    using ExecuteMotion = roboparty_inference::action::ExecuteMotion;
    using ExecuteMotionGoalHandle = rclcpp_action::ServerGoalHandle<ExecuteMotion>;

    std::shared_ptr<RobotInterface> robot_;
    std::atomic<bool> is_running_{false}, is_joy_control_{true}, is_interrupt_{false}, is_motion_policy_{false};
    std::string robot_config_path_;
    std::string perception_obs_topic_;
    size_t current_motion_policy_idx_ = 0;
    int active_policy_idx_ = 0;
    int locomotion_policy_idx_ = -1;
    int perception_obs_num_, joint_num_;
    bool use_depth_ = false;
    int decimation_;
    std::unique_ptr<Ort::Env> env_;
    int intra_threads_;
    Ort::AllocatorWithDefaultOptions allocator_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr perception_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr action_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr clear_depth_history_client_;
    std::thread inference_thread_;
    std::thread control_thread_;
    std::thread reset_thread_;
    std::thread motion_action_thread_;
    std::mutex reset_thread_mutex_;
    bool reset_thread_running_ = false;
    float act_alpha_;
    float dt_;
    float obs_scales_lin_vel_, obs_scales_ang_vel_, obs_scales_dof_pos_, obs_scales_dof_vel_,
        obs_scales_gravity_b_, clip_observations_;
    float clip_actions_;
    float action_rescale_ = 1.0f;
    double cmd_vel_timeout_s_ = 0.25;
    std::chrono::steady_clock::time_point last_external_cmd_time_{};
    bool external_cmd_seen_ = false;
    std::vector<double> action_scale_, clip_cmd_, joint_default_angle_, joint_limits_;
    std::vector<long int> usd2urdf_;
    float gravity_z_upper_;
    int last_button0_ = 0, last_button1_ = 0, last_button2_ = 0, last_button3_ = 0, last_button4_ = 0, last_button5_ = 0;
    std::vector<PolicyRuntime> policies_;
    std::vector<int> motion_policy_indices_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_joints_service_, set_zeros_service_, clear_errors_service_, refresh_joints_service_, read_joints_service_, read_imu_service_, init_motors_service_, deinit_motors_service_, start_inference_service_, stop_inference_service_;
    rclcpp::Service<roboparty_inference::srv::SetCommandSource>::SharedPtr set_command_source_service_;
    rclcpp::Publisher<roboparty_inference::msg::RuntimeState>::SharedPtr runtime_state_publisher_;
    rclcpp::TimerBase::SharedPtr runtime_state_timer_;
    rclcpp_action::Server<ExecuteMotion>::SharedPtr execute_motion_action_server_;
    std::atomic<bool> motion_action_active_{false};
    std::atomic<bool> motion_cancel_requested_{false};
    std::atomic<uint64_t> action_generation_{0};
    std::atomic<uint64_t> applied_action_generation_{0};
    rclcpp_action::GoalUUID active_motion_goal_uuid_{};
    std::string runtime_fault_;

    std::mutex act_mutex_, perception_mutex_, interrupt_mutex_, cmd_mutex_, mode_mutex_,
        control_mutex_, lb_switch_mutex_, lifecycle_mutex_, motion_goal_mutex_,
        runtime_fault_mutex_;
    std::vector<float> act_, last_act_, cmd_vel_, interrupt_action_, perception_obs_buffer_;
    std::vector<float> joint_pos_buffer_, joint_vel_buffer_, joint_torques_buffer_, quat_buffer_, ang_vel_buffer_;
    sensor_msgs::msg::JointState joint_state_msg_, action_msg_;

    void subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg);
    void subs_cmd_callback(const std::shared_ptr<geometry_msgs::msg::Twist> msg);
    void subs_perception_callback(const std::shared_ptr<std_msgs::msg::Float32MultiArray> msg);
    void subs_joint_state_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void inference();
    void control();
    void apply_action();
    bool start_joint_reset();
    PolicyRuntime& active_policy();

    void load_config();
    void setup_model(std::unique_ptr<ModelContext>& ctx, std::string model_path, int input_size);

    // Policy/model runtime helpers.
    void initialize_runtime_state();
    void reset_runtime_state();
    void request_depth_history_reset();
    void reset_policy_runtime(PolicyRuntime& policy);
    void step_motion_frame();
    void zero_cmd_vel();
    void request_motion_cancel();
    void set_runtime_fault(const std::string& fault);
    bool try_start_inference(std::string& reason);
    bool resume_inference_if_fault_free(std::string& reason);
    bool external_command_is_fresh();
    bool activate_motion_policy(
        const std::string& policy_id,
        const rclcpp_action::GoalUUID& goal_uuid,
        std::string& error);
    void return_to_locomotion(const std::string& reason);
    int find_policy_by_id(const std::string& policy_id) const;

    // Observation registry and layout helpers.
    static const std::vector<ObsSourceDefinition>& obs_source_definitions();
    std::vector<ObsSourceSpec> parse_obs_layout(const std::string& layout_spec,
                                                const std::string& layout_name);
    bool has_obs_source(const std::string& source_name) const;
    ObsStackOrder parse_obs_stack_order(const std::string& stack_order_name);

    // Observation runtime helpers.
    void update_obs_segments(std::vector<std::vector<float>>& segments,
                             const std::vector<ObsSourceSpec>& layout);
    void flatten_obs_segments(const std::vector<std::vector<float>>& segments,
                              std::vector<float>::iterator output_begin);
    void update_obs_history(std::vector<float>& history, const std::vector<float>& obs,
                            int obs_num, int frame_stack, bool is_first_frame);
    void update_stacked_obs(std::vector<float>& input_buffer, const std::vector<float>& obs,
                            int obs_num, int frame_stack, ObsStackOrder stack_order,
                            const std::vector<int>& field_sizes, bool is_first_frame);
    void gather_sparse_obs_history(std::vector<float>& input_buffer,
                                   const std::vector<float>& obs_history,
                                   const std::vector<ObsHistorySlice>& gather_plan);

    // Observation getters.
    void get_cmd_vel_obs(std::vector<float>& segment);
    void get_ang_vel_obs(std::vector<float>& segment);
    void get_gravity_b_obs(std::vector<float>& segment);
    void get_dof_pos_obs(std::vector<float>& segment);
    void get_dof_vel_obs(std::vector<float>& segment);
    void get_last_action_obs(std::vector<float>& segment);
    void get_interrupt_obs(std::vector<float>& segment);
    void get_perception_obs(std::vector<float>& segment);
    void get_motion_command_obs(std::vector<float>& segment);
    void get_latent_obs(std::vector<float>& segment);

    void init_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void deinit_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                           std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void reset_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void set_zeros_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                       std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void clear_errors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void refresh_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void read_joints_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void read_imu_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void start_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                             std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void stop_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void set_command_source_srv(
        const std::shared_ptr<roboparty_inference::srv::SetCommandSource::Request> request,
        std::shared_ptr<roboparty_inference::srv::SetCommandSource::Response> response);
    rclcpp_action::GoalResponse handle_motion_goal(
        const rclcpp_action::GoalUUID& uuid,
        std::shared_ptr<const ExecuteMotion::Goal> goal);
    rclcpp_action::CancelResponse handle_motion_cancel(
        const std::shared_ptr<ExecuteMotionGoalHandle> goal_handle);
    void handle_motion_accepted(const std::shared_ptr<ExecuteMotionGoalHandle> goal_handle);
    void execute_motion(const std::shared_ptr<ExecuteMotionGoalHandle> goal_handle);
    void finish_motion(
        const std::shared_ptr<ExecuteMotionGoalHandle> goal_handle,
        bool requested_success,
        const std::string& message,
        const std::string& return_reason) noexcept;
    void publish_runtime_state();
    void publish_terminal_fault_state(const std::string& fault);
    void publish_joint_states();
    void publish_action();
    void publish_imu();
    
    template <typename T>
    void print_vector(const std::string& name, const std::vector<T>& vec) {
        std::stringstream ss;
        ss << name << ": [";
        for (size_t i = 0; i < vec.size(); ++i) {
            ss << vec[i] << (i == vec.size() - 1 ? "" : ", ");
        }
        ss << "]";
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
    }
};
