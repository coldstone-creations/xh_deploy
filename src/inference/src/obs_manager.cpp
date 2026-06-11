#include "inference_node.hpp"

// ---------------------------------------------------------------------------
// Layout / stack-order helpers (thin wrappers)
// ---------------------------------------------------------------------------

ObsStackOrder InferenceNode::parse_obs_stack_order(const std::string& name) {
    if (name == "frame_major") return ObsStackOrder::FrameMajor;
    if (name == "obs_major")   return ObsStackOrder::ObsMajor;
    throw std::runtime_error("Unsupported obs stack order: " + name);
}

bool InferenceNode::has_obs_source(const std::string& source_name) const {
    for (size_t i = 0; i < policy_mgr_.count(); ++i) {
        const auto& cfg = policy_mgr_.config(i);
        const auto matches = [&source_name](const ObsSourceSpec& spec) {
            return spec.name == source_name;
        };
        if (std::any_of(cfg.obs_layout.begin(), cfg.obs_layout.end(), matches) ||
            std::any_of(cfg.extra_obs_layout.begin(), cfg.extra_obs_layout.end(), matches)) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Motion frame stepping
// ---------------------------------------------------------------------------

void InferenceNode::step_motion_frame() {
    policy_mgr_.step_motion_frame();
}

// ---------------------------------------------------------------------------
// Observation getters
// ---------------------------------------------------------------------------

void InferenceNode::get_motion_pos_obs(std::vector<float>& segment) {
    const size_t idx = policy_mgr_.active_index();
    auto loader = policy_mgr_.motion_loader(idx);
    if (!loader) return;
    const auto& st = policy_mgr_.active_state();
    const std::vector<float>& pos = loader->get_pos(st.motion_frame);
    std::copy(pos.begin(), pos.end(), segment.begin());
}

void InferenceNode::get_motion_vel_obs(std::vector<float>& segment) {
    const size_t idx = policy_mgr_.active_index();
    auto loader = policy_mgr_.motion_loader(idx);
    if (!loader) return;
    const auto& st = policy_mgr_.active_state();
    const std::vector<float>& vel = loader->get_vel(st.motion_frame);
    std::copy(vel.begin(), vel.end(), segment.begin());
}

void InferenceNode::get_ang_vel_obs(std::vector<float>& segment) {
    ang_vel_buffer_ = robot_->get_ang_vel();
    for (int i = 0; i < 3; i++) {
        segment[i] = ang_vel_buffer_[i] * obs_scales_ang_vel_;
    }
}

void InferenceNode::get_gravity_b_obs(std::vector<float>& segment) {
    quat_buffer_ = robot_->get_quat();
    Eigen::Quaternionf q_b2w(quat_buffer_[0], quat_buffer_[1], quat_buffer_[2], quat_buffer_[3]);
    Eigen::Vector3f gravity_w(0.0f, 0.0f, -1.0f);
    Eigen::Quaternionf q_w2b = q_b2w.inverse();
    Eigen::Vector3f gravity_b = q_w2b * gravity_w;
    safety_monitor_.check_fall_down(gravity_b.z());
    segment[0] = gravity_b.x() * obs_scales_gravity_b_;
    segment[1] = gravity_b.y() * obs_scales_gravity_b_;
    segment[2] = gravity_b.z() * obs_scales_gravity_b_;
}

void InferenceNode::get_cmd_vel_obs(std::vector<float>& segment) {
    std::unique_lock<std::mutex> lock(cmd_mutex_);
    segment[0] = cmd_vel_[0] * obs_scales_lin_vel_;
    segment[1] = cmd_vel_[1] * obs_scales_lin_vel_;
    segment[2] = cmd_vel_[2] * obs_scales_ang_vel_;
}

void InferenceNode::get_dof_pos_obs(std::vector<float>& segment) {
    joint_pos_buffer_ = robot_->get_joint_q();
    safety_monitor_.check_joint_limits(joint_pos_buffer_, usd2urdf_);
    for (int i = 0; i < joint_num_; i++) {
        segment[i] = (joint_pos_buffer_[usd2urdf_[i]] - joint_default_angle_[usd2urdf_[i]]) * obs_scales_dof_pos_;
    }
}

void InferenceNode::get_dof_vel_obs(std::vector<float>& segment) {
    joint_vel_buffer_ = robot_->get_joint_vel();
    for (int i = 0; i < joint_num_; i++) {
        segment[i] = joint_vel_buffer_[usd2urdf_[i]] * obs_scales_dof_vel_;
    }
}

void InferenceNode::get_last_action_obs(std::vector<float>& segment) {
    const auto& output = policy_mgr_.active_runner().output_buffer();
    for (int i = 0; i < joint_num_; i++) {
        segment[i] = output[i];
    }
}

void InferenceNode::get_interrupt_obs(std::vector<float>& segment) {
    segment[0] = is_interrupt_.load() ? 1.0f : 0.0f;
}

void InferenceNode::get_perception_obs(std::vector<float>& segment) {
    std::unique_lock<std::mutex> lock(perception_mutex_);
    std::copy(perception_obs_buffer_.begin(), perception_obs_buffer_.begin() + segment.size(), segment.begin());
}
