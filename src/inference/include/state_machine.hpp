#pragma once

#include "state_base.hpp"

#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <mutex>

/// @brief Headless state machine — owns states, orchestrates transitions.
///
/// Usage:
///   1. Create states via add_state().
///   2. Set initial state via set_initial().
///   3. Each tick: call Run(obs, cmd).
///   4. Safety monitor calls ForceTransition(kJointDamping) on violation.
///   5. Call joint_command() AFTER Run() to read the state's output.
class StateMachine {
public:
    StateMachine() = default;

    /// Register a state instance.  The first call to add_state also sets the
    /// initial state unless set_initial() is called afterwards.
    void add_state(std::shared_ptr<StateBase> state);

    /// Explicitly set which state to start from.
    void set_initial(StateName name);

    /// Execute one tick.
    /// @return the current state name after this tick (may have changed).
    StateName Run(const RobotObs& obs, const UserCommand& cmd);

    /// Force an immediate transition on the NEXT Run() call.
    /// Used by the safety monitor to override normal transitions.
    void ForceTransition(StateName target);

    /// Current state name.
    StateName current_state() const { return current_state_name_; }

    /// Look up a state by name.
    std::shared_ptr<StateBase> get_state(StateName name);

    /// Convenience: return the string name of the current state.
    const char* current_state_name_str() const;

private:
    std::map<StateName, std::shared_ptr<StateBase>> states_;
    std::shared_ptr<StateBase> current_state_;
    StateName current_state_name_ = StateName::kIdle;
    StateName forced_next_state_  = StateName::kIdle;
    bool      force_pending_      = false;
    bool      initial_set_        = false;
};

// ============================================================================
// Inline implementations
// ============================================================================

inline void StateMachine::add_state(std::shared_ptr<StateBase> state) {
    auto name = state->name();
    states_[name] = std::move(state);
    if (!initial_set_) {
        current_state_      = states_[name];
        current_state_name_ = name;
        initial_set_        = true;
    }
}

inline void StateMachine::set_initial(StateName name) {
    auto it = states_.find(name);
    if (it == states_.end()) {
        std::cerr << "[StateMachine] set_initial: unknown state\n";
        return;
    }
    current_state_      = it->second;
    current_state_name_ = name;
    initial_set_        = true;
}

inline StateName StateMachine::Run(const RobotObs& obs, const UserCommand& cmd) {
    // --- Apply forced transition if pending ---
    if (force_pending_) {
        force_pending_ = false;
        StateName target = forced_next_state_;
        if (target != current_state_name_) {
            auto it = states_.find(target);
            if (it != states_.end()) {
                current_state_->OnExit();
                std::cout << "[SM] " << current_state_name_str()
                          << " → " << StateNameToStr(it->second->name())
                          << " (forced)\n";
                current_state_      = it->second;
                current_state_name_ = target;
                current_state_->OnEnter();
            }
        }
        return current_state_name_;
    }

    // --- Normal execution ---
    current_state_->Run(obs, cmd);

    // --- Transition check ---
    StateName next = current_state_->GetNextStateName(obs, cmd);
    if (next != current_state_name_) {
        auto it = states_.find(next);
        if (it != states_.end()) {
            current_state_->OnExit();
            std::cout << "[SM] " << current_state_name_str()
                      << " → " << StateNameToStr(it->second->name()) << "\n";
            current_state_      = it->second;
            current_state_name_ = next;
            current_state_->OnEnter();
        }
    }

    return current_state_name_;
}

inline void StateMachine::ForceTransition(StateName target) {
    forced_next_state_ = target;
    force_pending_     = true;
}

inline std::shared_ptr<StateBase> StateMachine::get_state(StateName name) {
    auto it = states_.find(name);
    return (it != states_.end()) ? it->second : nullptr;
}

inline const char* StateMachine::current_state_name_str() const {
    switch (current_state_name_) {
        case StateName::kIdle:         return "Idle";
        case StateName::kStandUp:      return "StandUp";
        case StateName::kRLControl:    return "RLControl";
        case StateName::kJointDamping: return "JointDamping";
        default:                       return "Unknown";
    }
}
