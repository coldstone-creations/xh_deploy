#include "inference_node.hpp"

// ============================================================================
// Runtime state helpers
// ============================================================================

void InferenceNode::reset_runtime_state() {
    is_running_.store(false);
    is_interrupt_.store(false);
    is_motion_policy_.store(false);
    current_motion_policy_idx_ = 0;
    state_machine_.ForceTransition(StateName::kIdle);
    policy_mgr_.set_active(0);
    {
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        std::fill(cmd_vel_.begin(), cmd_vel_.end(), 0.0f);
    }
    {
        std::unique_lock<std::mutex> lock(perception_mutex_);
        std::fill(perception_obs_buffer_.begin(), perception_obs_buffer_.end(), 0.0f);
    }
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        for (int i = 0; i < joint_num_; i++) {
            act_[i]      = static_cast<float>(joint_default_angle_[i]);
            last_act_[i] = static_cast<float>(joint_default_angle_[i]);
        }
    }
    if (supports_interrupt()) {
        if (joint_default_angle_.size() < interrupt_action_.size()) {
            throw std::runtime_error("joint_default_angle is smaller than interrupt_action");
        }
        std::unique_lock<std::mutex> lock(interrupt_mutex_);
        const size_t offset = joint_default_angle_.size() - interrupt_action_.size();
        for (size_t i = 0; i < interrupt_action_.size(); i++) {
            interrupt_action_[i] = static_cast<float>(joint_default_angle_[offset + i]);
        }
    }
    policy_mgr_.reset_all();
}

void InferenceNode::initialize_runtime_state() {
    policy_mgr_.set_active(0);

    joint_state_msg_.name.resize(joint_num_);
    joint_state_msg_.position.assign(joint_num_, 0.0f);
    joint_state_msg_.velocity.assign(joint_num_, 0.0f);
    joint_state_msg_.effort.assign(joint_num_, 0.0f);
    action_msg_.name.resize(joint_num_);
    action_msg_.position.assign(joint_num_, 0.0f);
    for (int i = 0; i < joint_num_; i++) {
        joint_state_msg_.name[i] = "joint_" + std::to_string(i + 1);
        action_msg_.name[i]      = "action_" + std::to_string(i + 1);
    }

    cmd_vel_.assign(3, 0.0f);
    act_.assign(joint_num_, 0.0f);
    last_act_.assign(joint_num_, 0.0f);
    joint_pos_buffer_.assign(joint_num_, 0.0f);
    joint_vel_buffer_.assign(joint_num_, 0.0f);
    joint_torques_buffer_.assign(joint_num_, 0.0f);
    quat_buffer_.assign(4, 0.0f);
    ang_vel_buffer_.assign(3, 0.0f);
    if (has_obs_source("perception")) {
        perception_obs_buffer_.assign(perception_obs_num_, 0.0f);
    } else {
        perception_obs_buffer_.clear();
    }
    if (has_obs_source("interrupt")) {
        interrupt_action_.assign(10, 0.0f);
    } else {
        interrupt_action_.clear();
    }
}

// ============================================================================
// build_obs / build_cmd — gather sensor + user data for the state machine
// ============================================================================

void InferenceNode::build_obs(RobotObs& obs) {
    // Read from robot (these calls are mutex-protected inside RobotInterface)
    obs.joint_pos = robot_->get_joint_q();
    obs.joint_vel = robot_->get_joint_vel();
    obs.quat      = robot_->get_quat();
    obs.ang_vel   = robot_->get_ang_vel();

    // cmd_vel (thread-safe snapshot)
    {
        std::unique_lock<std::mutex> lock(cmd_mutex_);
        obs.cmd_vel = cmd_vel_;
    }

    // perception data
    {
        std::unique_lock<std::mutex> lock(perception_mutex_);
        obs.perception = perception_obs_buffer_;
    }

    // interrupt
    obs.interrupt_active = is_interrupt_.load();
    if (obs.interrupt_active) {
        std::unique_lock<std::mutex> lock(interrupt_mutex_);
        obs.interrupt_action = interrupt_action_;
    }

    // timestamp
    obs.time = this->now().seconds();
}

UserCommand InferenceNode::build_cmd() const {
    UserCommand cmd;
    cmd.joy_control = is_joy_control_.load();

    // Map state machine + joystick state to target_mode
    auto sm_state = state_machine_.current_state();
    if (is_motion_policy_.load()) {
        cmd.target_mode = static_cast<int>(StateName::kRLControl);  // sub-mode handled internally
    } else if (is_interrupt_.load()) {
        cmd.target_mode = static_cast<int>(StateName::kRLControl);
    } else {
        cmd.target_mode = static_cast<int>(sm_state);
    }
    return cmd;
}

// ============================================================================
// Control thread — reads act_ buffer, smooths, sends to robot
// ============================================================================

void InferenceNode::apply_action() {
    if (!robot_->is_init_.load()) return;

    // In Idle / StandUp / JointDamping, the state writes to act_ directly.
    // The control thread always runs the alpha filter + PD send.
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        for (size_t i = 0; i < act_.size(); i++) {
            last_act_[i] = act_alpha_ * act_[i] + (1 - act_alpha_) * last_act_[i];
        }
    }
    robot_->apply_action(last_act_);
}

void InferenceNode::control() {
    pthread_setname_np(pthread_self(), "control");
    struct sched_param sp{}; sp.sched_priority = 70;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(this->get_logger(), "Failed to set realtime priority for control thread");
        rclcpp::shutdown();
        return;
    }
    auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000000));
    while (rclcpp::ok()) {
        auto loop_start = std::chrono::steady_clock::now();
        try {
            apply_action();
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Exception in control thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }
        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed  = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        auto sleep_t  = period - elapsed;
        if (sleep_t > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleep_t);
        }
    }
}

// ============================================================================
// Inference / main-loop thread
// ============================================================================

void InferenceNode::inference() {
    pthread_setname_np(pthread_self(), "inference");
    struct sched_param sp{}; sp.sched_priority = 70;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(this->get_logger(), "Failed to set realtime priority for inference thread");
        rclcpp::shutdown();
        return;
    }
    auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000 * 1000 * decimation_));

    RobotObs   obs;
    UserCommand cmd;

    while (rclcpp::ok()) {
        auto loop_start = std::chrono::steady_clock::now();

        try {
            // 1. Gather sensor snapshot
            build_obs(obs);

            // 2. Gather user intent
            cmd = build_cmd();

            // 3. Run state machine — the active state does the work
            std::unique_lock<std::mutex> mode_lock(mode_mutex_);
            state_machine_.Run(obs, cmd);

            // 4. Safety: ask the current state if it should be forced to JointDamping
            {
                auto st = state_machine_.get_state(state_machine_.current_state());
                if (st && st->LoseControlJudge(obs)) {
                    RCLCPP_WARN(this->get_logger(), "LoseControlJudge triggered → JointDamping");
                    state_machine_.ForceTransition(StateName::kJointDamping);
                }
            }

            // 5. Publish (always — regardless of state)
            publish_imu();
            publish_joint_states();
            {
                std::unique_lock<std::mutex> lock(act_mutex_);
                publish_action();
            }

        } catch (const std::runtime_error& e) {
            // Safety violation → callback already called ForceTransition.
            // Just log and continue — the next tick will be in JointDamping.
            RCLCPP_WARN(this->get_logger(), "Safety catch: %s — transitioning to JointDamping", e.what());
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Fatal exception in inference thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed  = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        auto sleep_t  = period - elapsed;
        if (sleep_t > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleep_t);
        } else {
            RCLCPP_WARN(this->get_logger(),
                "Inference loop overran! Took %lld us, period=%lld us.",
                static_cast<long long>(elapsed.count()),
                static_cast<long long>(period.count()));
        }
    }
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        RCLCPP_WARN(rclcpp::get_logger("main"), "mlockall failed.");
    }
    pthread_setname_np(pthread_self(), "main");
    struct sched_param sp{}; sp.sched_priority = 50;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Failed to set realtime priority for main thread");
        rclcpp::shutdown();
        return 1;
    }
    std::shared_ptr<InferenceNode> node;
    try {
        node = std::make_shared<InferenceNode>();
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        RCLCPP_INFO(node->get_logger(), "=== Robot State Machine ===");
        RCLCPP_INFO(node->get_logger(), "  Idle → (A)StandUp → (B)RLControl → (B)Idle");
        RCLCPP_INFO(node->get_logger(), "  Any state → JointDamping (safety fallback, auto-return 3s)");
        RCLCPP_INFO(node->get_logger(), "Press 'X' to initialize/deinitialize motors");
        RCLCPP_INFO(node->get_logger(), "Press 'A' to stand up (cubic-spline trajectory)");
        RCLCPP_INFO(node->get_logger(), "Press 'B' to start/pause RL inference");
        RCLCPP_INFO(node->get_logger(), "Press 'Y' to switch Gamepad / cmd_vel control");
        if (node->supports_interrupt() || node->has_motion_policy()) {
            RCLCPP_INFO(node->get_logger(), "Press 'LB' to switch policy mode");
        }
        if (node->has_motion_policy()) {
            RCLCPP_INFO(node->get_logger(), "Press 'RB' to switch motion sequence");
        }
        executor.spin();
    } catch (const std::exception &e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Exception caught: %s", e.what());
    }
    rclcpp::shutdown();
    node.reset();
    return 0;
}
