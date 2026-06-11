#pragma once

#include "state_base.hpp"
#include <algorithm>
#include <iostream>

/// @brief JointDamping state — safety fallback with zero-torque damping.
///
/// Enters when any state's LoseControlJudge() returns true, or when the
/// safety monitor triggers a ForceTransition.  Holds for 3 seconds with
/// zero position targets (letting the robot's internal PD damping bring
/// it to a safe stop), then automatically returns to Idle.
class JointDampingState : public StateBase {
public:
    explicit JointDampingState(std::shared_ptr<StateContext> ctx)
        : StateBase(StateName::kJointDamping, std::move(ctx))
    {}

    void OnEnter() override {
        enter_time_ = -1.0;
        // Capture current joint positions as hold targets (soft damped hold)
        // Actually, we set to default angle for a safe neutral pose.
        // But better: hold current position to avoid sudden motion.
        hold_pos_.clear();
        ctx_->is_running->store(false);
        std::cerr << "[JointDamping] SAFETY STOP — entering damping mode.\n";
    }

    void OnExit() override {
        std::cout << "[JointDamping] Exiting damping mode.\n";
    }

    void Run(const RobotObs& obs, const UserCommand& /*cmd*/) override {
        // Lazy init on first tick
        if (enter_time_ < 0.0) {
            enter_time_ = obs.time;
            hold_pos_   = obs.joint_pos;  // hold current position
        }

        // Apply zero-velocity damping: set target = current position,
        // which means the PD controller applies only damping (kd * vel).
        // The robot naturally settles without driving anywhere.
        {
            std::lock_guard<std::mutex> lock(*ctx_->act_mutex);
            auto& act = *ctx_->act;
            if (!hold_pos_.empty()) {
                for (size_t i = 0; i < hold_pos_.size() && i < act.size(); ++i) {
                    act[i] = hold_pos_[i];
                }
            }
        }

        // Log periodically
        double elapsed = obs.time - enter_time_;
        if (static_cast<int>(elapsed) != last_logged_sec_) {
            last_logged_sec_ = static_cast<int>(elapsed);
            std::cout << "[JointDamping] " << (3 - last_logged_sec_)
                      << "s until auto-return to Idle...\n";
        }
    }

    bool LoseControlJudge(const RobotObs& /*obs*/) override {
        return false;  // already in the safest state
    }

    StateName GetNextStateName(const RobotObs& obs,
                               const UserCommand& /*cmd*/) override {
        // Auto-return to Idle after 3 seconds
        if (enter_time_ > 0.0 && (obs.time - enter_time_) >= 3.0) {
            std::cout << "[JointDamping] Timer expired → Idle.\n";
            return StateName::kIdle;
        }
        return StateName::kJointDamping;
    }

private:
    double              enter_time_ = -1.0;
    int                 last_logged_sec_ = -1;
    std::vector<float>  hold_pos_;
};
