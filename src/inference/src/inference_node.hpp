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
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include "utils/motion_loader.hpp"
#include <std_srvs/srv/trigger.hpp>
#include "robot_interface.hpp"
#include "safety_monitor.hpp"
#include "observation_assembler.hpp"
#include "model_runner.hpp"
#include "policy_manager.hpp"
#include "state_base.hpp"
#include "state_machine.hpp"
#include "idle_state.hpp"
#include "standup_state.hpp"
#include "rl_control_state.hpp"
#include "joint_damping_state.hpp"

class InferenceNode : public rclcpp::Node {
   public:
    InferenceNode() : Node("inference_node") {
        load_config();

        robot_ = std::make_shared<RobotInterface>(std::string(ROOT_DIR) + "config/robot.yaml");

        Ort::ThreadingOptions thread_opts;
        if (intra_threads_ > 0) {
            thread_opts.SetGlobalIntraOpNumThreads(intra_threads_);
        }
        env_ = std::make_unique<Ort::Env>(thread_opts, ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeInference");

        if (policy_mgr_.count() == 0) {
            throw std::runtime_error("At least one policy must be configured");
        }

        // --- Create ModelRunner + MotionLoader for each policy ---
        for (size_t i = 0; i < policy_mgr_.count(); i++) {
            const auto& cfg = policy_mgr_.config(i);
            if (cfg.has_motion) {
                auto loader = std::make_shared<MotionLoader>(cfg.motion_path);
                if (loader->get_num_frames() == 0) {
                    throw std::runtime_error("Motion file has no frames: " + cfg.motion_path);
                }
                if (loader->get_num_joints() != static_cast<size_t>(joint_num_)) {
                    throw std::runtime_error("Motion joint count mismatch: " + cfg.motion_path);
                }
                policy_mgr_.set_motion_loader(i, std::move(loader));
            }
            auto runner = std::make_unique<ModelRunner>(*env_, allocator_, joint_num_);
            runner->setup_model(cfg.model_path,
                                cfg.obs_num * cfg.frame_stack + cfg.extra_obs_num);
            policy_mgr_.set_runner(i, std::move(runner));
        }

        initialize_runtime_state();

        // --- Build StateContext (shared by all states) ---
        auto ctx = std::make_shared<StateContext>();
        ctx->robot               = robot_.get();
        ctx->safety              = &safety_monitor_;
        ctx->policy_mgr          = &policy_mgr_;
        ctx->obs_asm             = &obs_assembler_;
        ctx->joint_num           = joint_num_;
        ctx->joint_default_angle = &joint_default_angle_;
        ctx->usd2urdf            = &usd2urdf_;
        ctx->action_scale        = action_scale_;
        ctx->clip_actions        = clip_actions_;
        ctx->act_mutex           = &act_mutex_;
        ctx->act                 = &act_;
        ctx->is_running          = &is_running_;

        // --- Setup SafetyMonitor with state-machine callback ---
        safety_monitor_.configure(gravity_z_upper_, joint_limits_, [this]() {
            RCLCPP_FATAL(this->get_logger(), "Safety violation → JointDamping");
            state_machine_.ForceTransition(StateName::kJointDamping);
        });

        // --- Register observation fillers ---
        {
            ObservationAssembler::StandardFillers f;
            f.cmd_vel     = [this](std::vector<float>& s) { get_cmd_vel_obs(s); };
            f.ang_vel     = [this](std::vector<float>& s) { get_ang_vel_obs(s); };
            f.gravity_b   = [this](std::vector<float>& s) { get_gravity_b_obs(s); };
            f.dof_pos     = [this](std::vector<float>& s) { get_dof_pos_obs(s); };
            f.dof_vel     = [this](std::vector<float>& s) { get_dof_vel_obs(s); };
            f.last_action = [this](std::vector<float>& s) { get_last_action_obs(s); };
            f.interrupt   = [this](std::vector<float>& s) { get_interrupt_obs(s); };
            f.perception  = [this](std::vector<float>& s) { get_perception_obs(s); };
            f.motion_pos  = [this](std::vector<float>& s) { get_motion_pos_obs(s); };
            f.motion_vel  = [this](std::vector<float>& s) { get_motion_vel_obs(s); };
            obs_assembler_.register_standard_fillers(f);
        }

        // --- Build state machine ---
        state_machine_.add_state(std::make_shared<IdleState>(ctx));
        state_machine_.add_state(std::make_shared<StandUpState>(ctx));
        auto rl_state = std::make_shared<RLControlState>(ctx);
        rl_state->set_clip_observations(clip_observations_);
        state_machine_.add_state(std::move(rl_state));
        state_machine_.add_state(std::make_shared<JointDampingState>(ctx));
        state_machine_.set_initial(StateName::kIdle);

        reset_runtime_state();

        // --- ROS communication ---
        auto data_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", data_qos, std::bind(&InferenceNode::subs_joy_callback, this, std::placeholders::_1));
        cmd_subscription_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", data_qos, std::bind(&InferenceNode::subs_cmd_callback,this, std::placeholders::_1
        ));
        elevation_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            perception_obs_topic_, data_qos,
            std::bind(&InferenceNode::subs_elevation_callback, this, std::placeholders::_1));
        joint_state_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_ref_states", data_qos,
            std::bind(&InferenceNode::subs_joint_state_callback, this, std::placeholders::_1));
        action_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/action", data_qos);
        imu_publisher_ =
            this->create_publisher<sensor_msgs::msg::Imu>("/imu", data_qos);
        joint_state_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", data_qos);

        inference_thread_ = std::thread(&InferenceNode::inference, this);
        control_thread_   = std::thread(&InferenceNode::control, this);

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
    }

    ~InferenceNode() {
        if (inference_thread_.joinable()) inference_thread_.join();
        if (control_thread_.joinable())   control_thread_.join();
        reset_runtime_state();
        if (robot_) robot_.reset();
    }

    bool supports_interrupt() const { return policy_mgr_.has_interrupt_source(); }
    bool has_motion_policy()   const { return policy_mgr_.has_motion_policy(); }

   private:
    // ========================================================================
    // Modular components
    // ========================================================================
    std::shared_ptr<RobotInterface> robot_;
    SafetyMonitor         safety_monitor_;
    ObservationAssembler  obs_assembler_;
    PolicyManager         policy_mgr_;
    StateMachine          state_machine_;

    // ========================================================================
    // Runtime state (progressively moving into StateMachine / StateContext)
    // ========================================================================
    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_joy_control_{true};
    std::atomic<bool> is_interrupt_{false};
    std::atomic<bool> is_motion_policy_{false};
    size_t current_motion_policy_idx_ = 0;

    std::string perception_obs_topic_;
    int perception_obs_num_, joint_num_;
    int decimation_;
    std::unique_ptr<Ort::Env> env_;
    int intra_threads_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // ========================================================================
    // ROS communication
    // ========================================================================
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr elevation_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr action_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    std::thread inference_thread_;
    std::thread control_thread_;

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
        reset_joints_service_, set_zeros_service_, clear_errors_service_,
        refresh_joints_service_, read_joints_service_, read_imu_service_,
        init_motors_service_, deinit_motors_service_,
        start_inference_service_, stop_inference_service_;

    // ========================================================================
    // Config + buffers
    // ========================================================================
    float act_alpha_, dt_;
    float obs_scales_lin_vel_, obs_scales_ang_vel_, obs_scales_dof_pos_,
          obs_scales_dof_vel_, obs_scales_gravity_b_, clip_observations_;
    float action_scale_, clip_actions_;
    std::vector<double> clip_cmd_, joint_default_angle_, joint_limits_;
    std::vector<long int> usd2urdf_;
    float gravity_z_upper_;
    int last_button0_ = 0, last_button1_ = 0, last_button2_ = 0,
        last_button3_ = 0, last_button4_ = 0, last_button5_ = 0;

    std::mutex act_mutex_, perception_mutex_, interrupt_mutex_,
               cmd_mutex_, mode_mutex_;
    std::vector<float> act_, last_act_, cmd_vel_, interrupt_action_, perception_obs_buffer_;
    std::vector<float> joint_pos_buffer_, joint_vel_buffer_, joint_torques_buffer_,
                       quat_buffer_, ang_vel_buffer_;
    sensor_msgs::msg::JointState joint_state_msg_, action_msg_;

    // ========================================================================
    // Methods
    // ========================================================================
    void subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg);
    void subs_cmd_callback(const std::shared_ptr<geometry_msgs::msg::Twist> msg);
    void subs_elevation_callback(const std::shared_ptr<std_msgs::msg::Float32MultiArray> msg);
    void subs_joint_state_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void inference();
    void control();
    void apply_action();

    void load_config();
    void initialize_runtime_state();
    void reset_runtime_state();

    bool has_obs_source(const std::string& source_name) const;
    static ObsStackOrder parse_obs_stack_order(const std::string& name);

    // Observation getters
    void get_cmd_vel_obs(std::vector<float>& segment);
    void get_ang_vel_obs(std::vector<float>& segment);
    void get_gravity_b_obs(std::vector<float>& segment);
    void get_dof_pos_obs(std::vector<float>& segment);
    void get_dof_vel_obs(std::vector<float>& segment);
    void get_last_action_obs(std::vector<float>& segment);
    void get_interrupt_obs(std::vector<float>& segment);
    void get_perception_obs(std::vector<float>& segment);
    void get_motion_pos_obs(std::vector<float>& segment);
    void get_motion_vel_obs(std::vector<float>& segment);

    // Services
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
    void publish_joint_states();
    void publish_action();
    void publish_imu();

    // Helpers
    void build_obs(RobotObs& obs);
    UserCommand build_cmd() const;

    template <typename T>
    void print_vector(const std::string& name, const std::vector<T>& vec) {
        std::stringstream ss;
        ss << name << ": [";
        for (size_t i = 0; i < vec.size(); ++i)
            ss << vec[i] << (i == vec.size() - 1 ? "" : ", ");
        ss << "]";
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
    }
};
