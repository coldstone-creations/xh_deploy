#pragma once

#include "state_base.hpp"
#include "observation_assembler.hpp"
#include "policy_manager.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>

/// @brief RLControl state — runs the policy inference pipeline.
///
/// Supports three sub-modes (toggled via UserCommand::target_mode):
///   - Normal:    default policy (index 0)
///   - Interrupt: joint mirrors from /joint_ref_states, remaining joints RL
///   - Motion:    motion-playback policy selected by PolicyManager
///
/// Internal sub-mode is controlled by the joystick LB/RB buttons, which
/// are translated to UserCommand::target_mode by InferenceNode.
class RLControlState : public StateBase {
public:
    explicit RLControlState(std::shared_ptr<StateContext> ctx)
        : StateBase(StateName::kRLControl, std::move(ctx))
    {}

    void OnEnter() override {
        std::cout << "[RLControl] Policy inference started (policy: "
                  << ctx_->policy_mgr->active_config().name << ").\n";
        ctx_->is_running->store(true);
    }

    void OnExit() override {
        ctx_->is_running->store(false);
        std::cout << "[RLControl] Policy inference paused.\n";
    }

    void Run(const RobotObs& obs, const UserCommand& /*cmd*/) override {
        auto& pm = *ctx_->policy_mgr;
        auto& oa = *ctx_->obs_asm;

        // --- Read fresh sensor data into assembler ---
        // (The getters registered in obs_assembler_ use robot_ directly,
        //  so assemble() will pull the latest data.  We don't need to
        //  pre-set each sensor value — the filler lambdas do that.)
        // However, for cmd_vel and perception which are NOT read from robot_,
        // we need to inject them.  These are handled by the existing getter
        // lambdas via InferenceNode's member variables (cmd_vel_, etc.).

        // --- Assemble observations ---
        auto& st  = pm.active_state();
        const auto& cfg = pm.active_config();
        auto& runner = pm.active_runner();

        oa.assemble(st.obs_segments, cfg.obs_layout);
        ObservationAssembler::flatten(st.obs_segments, st.obs.begin());

        // Clip observations
        for (auto& v : st.obs) {
            v = std::clamp(v, -clip_obs_, clip_obs_);
        }

        // Frame stacking
        ObservationAssembler::stack_frames(
            runner.input_buffer(), st.obs,
            cfg.obs_num, cfg.frame_stack, cfg.stack_order,
            cfg.obs_layout_sizes, st.is_first_frame);

        // Extra observations
        if (cfg.extra_obs_num > 0) {
            oa.assemble(st.extra_obs_segments, cfg.extra_obs_layout);
            ObservationAssembler::flatten(
                st.extra_obs_segments,
                runner.input_buffer().begin() + cfg.frame_stack * cfg.obs_num);
        }

        // Motion frame stepping
        if (cfg.has_motion) {
            pm.step_motion_frame();
        }
        st.is_first_frame = false;

        // --- ONNX Inference ---
        runner.run();

        // --- Write output actions ---
        {
            std::lock_guard<std::mutex> lock(*ctx_->act_mutex);
            auto& act = *ctx_->act;
            auto& output = runner.output_buffer();
            const auto& usd2urdf = *ctx_->usd2urdf;
            const auto& default_angle = *ctx_->joint_default_angle;

            for (size_t i = 0; i < output.size(); ++i) {
                output[i] = std::clamp(output[i], -ctx_->clip_actions,
                                       ctx_->clip_actions);
                int idx = usd2urdf[i];
                act[idx] = output[i] * ctx_->action_scale
                           + static_cast<float>(default_angle[idx]);
            }

            // Interrupt sub-mode: overwrite tail joints with external targets
            if (obs.interrupt_active && !obs.interrupt_action.empty()) {
                size_t offset = act.size() - obs.interrupt_action.size();
                for (size_t i = 0; i < obs.interrupt_action.size(); ++i) {
                    act[offset + i] = obs.interrupt_action[i];
                }
            }
        }
    }

    bool LoseControlJudge(const RobotObs& obs) override {
        // Delegates to SafetyMonitor — joint limits and fall-down
        if (ctx_->safety) {
            // Joint limits are checked inside get_dof_pos_obs (via safety_monitor_)
            // so a violation will throw.  We also do a posture check here.
        }

        // Posture check (same as Lite3): roll > 30°, pitch > 45°
        if (obs.quat.size() >= 4) {
            float qw = obs.quat[0], qx = obs.quat[1],
                  qy = obs.quat[2], qz = obs.quat[3];
            // Roll  ≈ atan2(2*(qw*qx + qy*qz), 1 - 2*(qx² + qy²))
            // Pitch ≈ asin(2*(qw*qy - qz*qx))
            float roll  = std::atan2(2.f*(qw*qx + qy*qz),
                                     1.f - 2.f*(qx*qx + qy*qy));
            float pitch = std::asin(std::clamp(2.f*(qw*qy - qz*qx), -1.f, 1.f));
            if (std::fabs(roll)  > 30.f * M_PI / 180.f ||
                std::fabs(pitch) > 45.f * M_PI / 180.f) {
                std::cerr << "[RLControl] POSTURE UNSAFE! roll="
                          << (180.f/M_PI)*roll << "° pitch="
                          << (180.f/M_PI)*pitch << "°\n";
                return true;
            }
        }
        return false;
    }

    StateName GetNextStateName(const RobotObs& /*obs*/,
                               const UserCommand& cmd) override {
        // Transition to Idle on B button (pause)
        if (cmd.target_mode == static_cast<int>(StateName::kIdle)) {
            return StateName::kIdle;
        }
        // Stay in RLControl (sub-mode switches are handled internally
        // via UserCommand::target_mode and joystick callbacks)
        return StateName::kRLControl;
    }

    void set_clip_observations(float v) { clip_obs_ = v; }

private:
    float clip_obs_ = 100.0f;
};
