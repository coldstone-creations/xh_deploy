/**
 * @file state_machine.hpp
 * @brief Pure-computation state machine — NO I/O, NO threading.
 *
 * This is a HEADLESS state machine: it receives observations and user commands
 * from the caller (ControlThread) and returns the next state + joint commands.
 * All sensor I/O and thread management lives in controller_core.hpp.
 *
 * Key differences from the original (Lite3_rl_deploy-main):
 *   1. Run() accepts external obs/cmd — no direct robot_/user_cmd_ access.
 *   2. ForceTransition() allows the safety monitor to override.
 *   3. State subclasses receive obs/cmd as function parameters.
 *
 * @date 2025-08-10
 */

 #pragma once

 #include "state_base.h"
 #include "idle_state.hpp"
 #include "standup_state.hpp"
 #include "joint_damping_state.hpp"
 #include "rl_control_state.hpp"
 
 #include <map>
 #include <memory>
 #include <iostream>
 
 class StateMachine {
 public:
     StateMachine(RobotType robot_type,
                  ControlParameters& cp,
                  DataStreaming& ds,
                  std::shared_ptr<PolicyRunnerBase> policy);
 
     /**
      * @brief Execute one tick of the state machine.
      *
      * @param obs  Latest sensor observation (already populated by caller).
      * @param cmd  Latest user command.
      * @return     The next StateName (may differ from current if a transition
      *             just occurred).
      */
     StateName Run(const RobotBasicState& obs, const UserCommand& cmd);
 
     /**
      * @brief Force an immediate transition — used by the safety monitor.
      *
      * On the NEXT call to Run(), the machine will enter the target state
      * regardless of what the current state's GetNextStateName() returns.
      */
     void ForceTransition(StateName target);
 
     /// Current state name (read-only).
     StateName CurrentState() const { return current_state_name_; }
 
     /// Access the policy runner for manual action retrieval.
     std::shared_ptr<PolicyRunnerBase> GetPolicy() { return policy_ptr_; }
 
     /// Access the data streaming instance.
     DataStreaming& GetDataStreaming() { return ds_; }
 
     /// Access the CURRENT state's joint command output buffer.
     /// Call this AFTER Run() to read what the state wants to send to the robot.
     MatXf& GetJointCommand() { return current_controller_->joint_cmd_output_; }
 
     /// Look up a state by name (public — used by ControlLoop for debug/logging).
     std::shared_ptr<StateBase> GetStatePtr(StateName name);
 
 private:
     // State instances
     std::shared_ptr<StateBase> idle_controller_;
     std::shared_ptr<StateBase> standup_controller_;
     std::shared_ptr<StateBase> rl_controller_;
     std::shared_ptr<StateBase> joint_damping_controller_;
 
     std::shared_ptr<StateBase> current_controller_;
     StateName current_state_name_;
     StateName forced_next_state_;      // set by ForceTransition(); kInvalid = auto
 
     ControlParameters& cp_;
     DataStreaming& ds_;
     std::shared_ptr<PolicyRunnerBase> policy_ptr_;
 };
 
 // ============================================================================
 // Implementation (inline for header-only convenience)
 // ============================================================================
 
 inline StateMachine::StateMachine(RobotType robot_type,
                                   ControlParameters& cp,
                                   DataStreaming& ds,
                                   std::shared_ptr<PolicyRunnerBase> policy)
     : current_state_name_(StateName::kIdle),
       forced_next_state_(StateName::kInvalid),
       cp_(cp), ds_(ds), policy_ptr_(policy)
 {
     // Build ControllerData (no robot/user_cmd pointers — state machine is headless)
     auto data = std::make_shared<ControllerData>();
     data->cp_ptr = std::make_shared<ControlParameters>(cp);  // share
     data->ds_ptr = std::make_shared<DataStreaming>(ds);       // share
 
     idle_controller_          = std::make_shared<IdleState>(robot_type, "idle", data);
     standup_controller_       = std::make_shared<StandUpState>(robot_type, "standup", data);
     rl_controller_            = std::make_shared<RLControlStateV2>(robot_type, "rl_control", data, policy);
     joint_damping_controller_ = std::make_shared<JointDampingState>(robot_type, "joint_damping", data);
 
     current_controller_ = idle_controller_;
 }
 
 inline StateName StateMachine::Run(const RobotBasicState& obs, const UserCommand& cmd) {
     // --- Apply forced transition if requested ---
     if (forced_next_state_ != StateName::kInvalid) {
         StateName target = forced_next_state_;
         forced_next_state_ = StateName::kInvalid;
         if (target != current_state_name_) {
             current_controller_->OnExit();
             current_controller_ = GetStatePtr(target);
             current_controller_->OnEnter();
             current_state_name_ = target;
         }
         return current_state_name_;
     }
 
     // --- Normal state execution ---
     current_controller_->Run(obs, cmd);
 
     // --- Check for state transition ---
     StateName next = current_controller_->GetNextStateName(obs, cmd);
     if (next != current_state_name_) {
         current_controller_->OnExit();
         std::cout << "[SM] " << current_controller_->state_name_
                   << " → " << GetStatePtr(next)->state_name_ << "\n";
         current_controller_ = GetStatePtr(next);
         current_controller_->OnEnter();
         current_state_name_ = next;
     }
 
     return current_state_name_;
 }
 
 inline void StateMachine::ForceTransition(StateName target) {
     forced_next_state_ = target;
 }
 
 inline std::shared_ptr<StateBase> StateMachine::GetStatePtr(StateName name) {
     switch (name) {
         case StateName::kInvalid:      return nullptr;
         case StateName::kIdle:         return idle_controller_;
         case StateName::kStandUp:      return standup_controller_;
         case StateName::kRLControl:    return rl_controller_;
         case StateName::kJointDamping: return joint_damping_controller_;
         default: return nullptr;
     }
 }
 