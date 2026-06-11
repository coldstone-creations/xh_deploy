#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <cstring>

// Forward declarations — no ROS / ONNX headers in this file.
class RobotInterface;
class SafetyMonitor;
class PolicyManager;
class ObservationAssembler;
class ModelRunner;

// ============================================================================
// StateName — all possible robot states
// ============================================================================
enum class StateName {
    kIdle,           // Motors initialised, waiting for stand-up command
    kStandUp,        // Cubic-spline transition to default standing pose
    kRLControl,      // Policy inference running (may have sub-modes)
    kJointDamping,   // Safety fallback — zero torque, damping only
};

// ============================================================================
// RobotObs — sensor snapshot passed into State::Run()
// ============================================================================
struct RobotObs {
    std::vector<float> joint_pos;
    std::vector<float> joint_vel;
    std::vector<float> quat;         // w, x, y, z  (body → world)
    std::vector<float> ang_vel;      // body-frame angular velocity
    std::vector<float> cmd_vel;      // [lin_x, lin_y, ang_z]  scaled
    std::vector<float> perception;   // elevation / height-map data
    std::vector<float> interrupt_action;  // external joint targets from /joint_ref_states
    bool    interrupt_active = false;
    double  time = 0.0;             // monotonic seconds (e.g. from rclcpp::Clock)
};

// ============================================================================
// UserCommand — user intent, fed into State::Run()
// ============================================================================
struct UserCommand {
    int  target_mode = 0;   // cast from StateName — non-zero = explicit mode request
    bool joy_control = true;
};

// ============================================================================
// StateContext — dependency bundle shared across all states
// ============================================================================
struct StateContext {
    // Hardware
    RobotInterface*       robot = nullptr;

    // Modular components (non-owning pointers)
    SafetyMonitor*        safety     = nullptr;
    PolicyManager*        policy_mgr = nullptr;
    ObservationAssembler* obs_asm    = nullptr;

    // Config
    int                   joint_num        = 0;
    const std::vector<double>* joint_default_angle = nullptr;
    const std::vector<int>*    usd2urdf           = nullptr;
    float action_scale   = 1.0f;
    float clip_actions   = 100.0f;

    // Shared output mutex + buffer (states WRITE, control thread READS)
    std::mutex*           act_mutex  = nullptr;
    std::vector<float>*   act        = nullptr;   // size = joint_num
    std::atomic<bool>*    is_running = nullptr;   // keeps control thread alive
};

// ============================================================================
// StateBase — pure-computation state (headless: no direct I/O)
// ============================================================================
class StateBase {
public:
    StateBase(StateName name, std::shared_ptr<StateContext> ctx)
        : state_name_(name), ctx_(std::move(ctx)) {}

    virtual ~StateBase() = default;

    // ---- Lifecycle ----
    virtual void OnEnter() = 0;
    virtual void OnExit()  = 0;

    /// Run one tick.  @p obs and @p cmd are provided by the main loop.
    virtual void Run(const RobotObs& obs, const UserCommand& cmd) = 0;

    /// @return true if the robot should be forced into JointDamping.
    virtual bool LoseControlJudge(const RobotObs& obs) = 0;

    /// @return the next state name (may equal current to stay).
    virtual StateName GetNextStateName(const RobotObs& obs,
                                       const UserCommand& cmd) = 0;

    // ---- Accessors ----
    StateName            name() const { return state_name_; }
    StateContext&        ctx()        { return *ctx_; }
    const StateContext&  ctx() const  { return *ctx_; }

protected:
    StateName                    state_name_;
    std::shared_ptr<StateContext> ctx_;
};
