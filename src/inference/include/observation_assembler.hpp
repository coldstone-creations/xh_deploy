#pragma once

#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <cctype>

/// Callback signature for filling one observation segment.
/// The segment vector is pre-sized to the declared field size;
/// the filler writes the current observation values into it.
using ObsFiller = std::function<void(std::vector<float>& segment)>;

/// Descriptor for one field in an observation layout.
struct ObsSourceSpec {
    std::string name;
    int         size = 0;
};

/// Stacking order for multi-frame observations.
enum class ObsStackOrder {
    FrameMajor,   ///< [frame0_obs0, frame0_obs1, ..., frame1_obs0, frame1_obs1, ...]
    ObsMajor,     ///< [obs0_frame0, obs0_frame1, ..., obs1_frame0, obs1_frame1, ...]
};

/// @brief Stateless assembler that builds observation vectors from named sources.
///
/// Usage:
///   1. Register fillers with register_filler("dof_pos", lambda).
///   2. Parse layout strings with parse_layout().
///   3. Each inference cycle:
///        a. Inject latest sensor data via set_*() helpers (or let fillers
///           capture external pointers).
///        b. Call assemble() to fill the per-layout segments.
///        c. Call stack_frames() to push segments into the flat input buffer.
class ObservationAssembler {
public:
    ObservationAssembler() = default;

    // ------------------------------------------------------------------
    // Filler registry
    // ------------------------------------------------------------------

    /// Register a named observation filler. Replaces any previous registration.
    void register_filler(const std::string& name, ObsFiller filler) {
        fillers_[name] = std::move(filler);
    }

    /// Return true if a filler with the given name is registered.
    bool has_filler(const std::string& name) const {
        return fillers_.find(name) != fillers_.end();
    }

    // ------------------------------------------------------------------
    // Layout parsing (static — does not depend on instance state)
    // ------------------------------------------------------------------

    /// Parse a comma-separated layout string "name:size,name:size,...".
    /// Format validation only; source-name existence is checked at assemble() time.
    /// @throws std::runtime_error on malformed input.
    static std::vector<ObsSourceSpec> parse_layout(
        const std::string& layout_spec,
        const std::string& layout_name);

    // ------------------------------------------------------------------
    // Observation assembly
    // ------------------------------------------------------------------

    /// Fill every segment in @p segments using the fillers listed in @p layout.
    /// @p segments must already be sized to match layout.size(), with each
    /// inner vector pre-allocated to the declared field size.
    void assemble(std::vector<std::vector<float>>& segments,
                  const std::vector<ObsSourceSpec>& layout) const;

    /// Flatten 2-D segments into a contiguous output range.
    static void flatten(const std::vector<std::vector<float>>& segments,
                        std::vector<float>::iterator output_begin);

    // ------------------------------------------------------------------
    // Frame stacking
    // ------------------------------------------------------------------

    /// Push the current flat observation into a multi-frame input buffer.
    /// Supports two stacking orders: FrameMajor and ObsMajor.
    /// @param input_buffer  The full stacked buffer (pre-allocated).
    /// @param obs           The current single-frame observation.
    /// @param obs_num       Number of floats in one observation frame.
    /// @param frame_stack   Number of frames to stack.
    /// @param stack_order   FrameMajor or ObsMajor.
    /// @param field_sizes   Per-field sizes (only used for ObsMajor).
    /// @param is_first_frame If true, replicate the first observation across
    ///                       all frames instead of shifting.
    static void stack_frames(std::vector<float>& input_buffer,
                             const std::vector<float>& obs,
                             int obs_num,
                             int frame_stack,
                             ObsStackOrder stack_order,
                             const std::vector<int>& field_sizes,
                             bool is_first_frame);

    // ------------------------------------------------------------------
    // Helpers for building a common set of fillers
    // ------------------------------------------------------------------

    /// Convenience: register the standard 10 observation sources in one call.
    /// The caller provides lambdas that capture the necessary state/sensors.
    struct StandardFillers {
        ObsFiller cmd_vel;
        ObsFiller ang_vel;
        ObsFiller gravity_b;
        ObsFiller dof_pos;
        ObsFiller dof_vel;
        ObsFiller last_action;
        ObsFiller interrupt;
        ObsFiller perception;
        ObsFiller motion_pos;
        ObsFiller motion_vel;
    };

    void register_standard_fillers(const StandardFillers& f);

private:
    std::unordered_map<std::string, ObsFiller> fillers_;
};
