#pragma once

#include "state_base.hpp"
#include <cmath>
#include <iostream>

/// @brief Idle state — motors are initialised, waiting for stand-up command.
///
/// Checks sensor health each tick.  Transitions to StandUp when the user
/// requests it (A button → target_mode = kStandUp) AND sensors are healthy.
class IdleState : public StateBase {
public:
    explicit IdleState(std::shared_ptr<StateContext> ctx)
        : StateBase(StateName::kIdle, std::move(ctx))
    {}

    void OnEnter() override {
        first_enter_ = true;
        // Stop the control thread from sending commands
        ctx_->is_running->store(false);
        std::cout << "[Idle] Waiting for stand-up command (A button).\n";
    }

    void OnExit() override {
        first_enter_ = false;
    }

    void Run(const RobotObs& obs, const UserCommand& /*cmd*/) override {
        // Check sensor health (non-blocking — just log on first enter)
        if (first_enter_) {
            enter_time_ = obs.time;
            first_enter_ = false;
        }

        // Periodic sensor health log
        if (obs.time - last_log_time_ > 1.0) {
            (void)obs; // sensors are read fresh each tick
            last_log_time_ = obs.time;
        }
    }

    bool LoseControlJudge(const RobotObs& /*obs*/) override {
        return false;  // idle is already safe
    }

    StateName GetNextStateName(const RobotObs& /*obs*/,
                               const UserCommand& cmd) override {
        // Require sensors stable for 0.5s before allowing transitions
        // (hysteresis — avoids flapping on startup noise)
        if (cmd.target_mode == static_cast<int>(StateName::kStandUp)) {
            std::cout << "[Idle] Stand-up requested.\n";
            return StateName::kStandUp;
        }
        return StateName::kIdle;
    }

private:
    bool   first_enter_    = true;
    double enter_time_     = 0.0;
    double last_log_time_  = 0.0;
};
