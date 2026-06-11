#pragma once

#include "observation_assembler.hpp"   // ObsSourceSpec, ObsStackOrder
#include "model_runner.hpp"
#include "utils/motion_loader.hpp"

#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

/// Immutable configuration for one policy.
struct PolicyConfig {
    std::string              name;
    std::string              model_path;
    std::string              motion_path;
    std::vector<ObsSourceSpec> obs_layout;
    std::vector<int>           obs_layout_sizes;
    int                      obs_num = 0;
    std::vector<ObsSourceSpec> extra_obs_layout;
    int                      extra_obs_num = 0;
    int                      frame_stack = 1;
    ObsStackOrder            stack_order = ObsStackOrder::FrameMajor;
    bool                     has_motion = false;
};

/// Mutable per-policy runtime state (reset on policy switch / restart).
struct PolicyRuntimeState {
    std::vector<std::vector<float>> obs_segments;
    std::vector<float>              obs;
    std::vector<std::vector<float>> extra_obs_segments;
    size_t                          motion_frame   = 0;
    bool                            is_first_frame = true;
};

/// Owns the set of policies and their runtime state.
///
/// Each policy also owns a ModelRunner (the ONNX session) and optionally a
/// MotionLoader.  Policy switching (interrupt / motion modes) is managed here.
class PolicyManager {
public:
    PolicyManager() = default;

    /// Append a policy from its parsed config.  Call once per policy during
    /// initialisation (before any runtime access).
    void add_policy(PolicyConfig cfg);

    // --- Queries -----------------------------------------------------------

    size_t count() const { return configs_.size(); }

    const PolicyConfig& config(size_t idx) const { return configs_[idx]; }
    PolicyRuntimeState& state(size_t idx)       { return states_[idx]; }

    size_t active_index() const { return active_idx_; }
    const PolicyConfig& active_config() const { return configs_[active_idx_]; }
    PolicyRuntimeState& active_state()         { return states_[active_idx_]; }

    ModelRunner& active_runner();
    ModelRunner& runner(size_t idx);

    /// True when any policy references a motion file.
    bool has_motion_policy() const { return !motion_indices_.empty(); }

    /// True when any policy references the "interrupt" observation source.
    bool has_interrupt_source() const { return has_interrupt_; }

    // --- Policy switching --------------------------------------------------

    void set_active(size_t idx);
    void reset_policy(size_t idx);
    void reset_all();

    /// Advance the motion frame of the active policy (no-op if no motion).
    void step_motion_frame();

    // --- Runner & motion setup (called during init) -----------------------

    /// Take ownership of a ModelRunner for policy @p idx.
    void set_runner(size_t idx, std::unique_ptr<ModelRunner> runner);

    /// Store a MotionLoader for policy @p idx.
    void set_motion_loader(size_t idx, std::shared_ptr<MotionLoader> loader);

    /// Access the MotionLoader for policy @p idx (may be null).
    std::shared_ptr<MotionLoader> motion_loader(size_t idx) const;

    // --- Motion helpers ----------------------------------------------------

    const std::vector<size_t>& motion_policy_indices() const { return motion_indices_; }
    size_t motion_policy_count() const { return motion_indices_.size(); }

private:
    std::vector<PolicyConfig>        configs_;
    std::vector<PolicyRuntimeState>  states_;
    std::vector<std::unique_ptr<ModelRunner>> runners_;
    std::vector<std::shared_ptr<MotionLoader>> motions_;
    std::vector<size_t>             motion_indices_;
    size_t                          active_idx_ = 0;
    bool                            has_interrupt_ = false;
};
