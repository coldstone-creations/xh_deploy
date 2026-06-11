#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

/// @brief Centralized safety checks: joint-limit violations and fall-down detection.
///
/// Instantiated by InferenceNode with callbacks that trigger rclcpp::shutdown().
/// All methods are non-virtual and trivial to inline — zero overhead in the
/// real-time inference path.
class SafetyMonitor {
public:
    using ShutdownCallback = std::function<void()>;

    /// Default-constructed; call configure() before use.
    SafetyMonitor() = default;

    /// @param gravity_z_upper  Maximum allowed gravity.z in body frame before
    ///                         declaring a fall-down.
    /// @param joint_limits     Flattened lower/upper pairs per joint (size =
    ///                         2 × joint_count).
    /// @param on_shutdown      Called when a violation is detected (typically
    ///                         logs the reason and calls rclcpp::shutdown()).
    void configure(float gravity_z_upper,
                   const std::vector<double>& joint_limits,
                   ShutdownCallback on_shutdown)
    {
        gravity_z_upper_ = gravity_z_upper;
        joint_limits_    = &joint_limits;
        on_shutdown_     = std::move(on_shutdown);
        configured_      = true;
    }

    /// Check every joint position against its configured limits.
    /// @return true if all joints are within limits; false if shutdown was
    ///         triggered (the caller should stop the current iteration).
    bool check_joint_limits(const std::vector<float>& joint_pos,
                            const std::vector<int>&   usd2urdf) const
    {
        (void)usd2urdf;  // kept for future use (e.g. remapping)
        const size_t joint_count = joint_limits_->size() / 2;
        for (size_t i = 0; i < joint_count; ++i) {
            const float  pos  = joint_pos[i];
            const double low  = (*joint_limits_)[i * 2];
            const double high = (*joint_limits_)[i * 2 + 1];
            if (pos < low || pos > high) {
                on_shutdown_();
                throw std::runtime_error(
                    "Joint " + std::to_string(i + 1) + " out of limit! "
                    "pos=" + std::to_string(pos) +
                    " limit=[" + std::to_string(low) + ", " + std::to_string(high) + "]");
            }
        }
        return true;
    }

    /// Check the body-frame gravity vector for fall-down.
    /// @param gravity_b_z  The z-component of gravity in the body frame.
    /// @return true if the robot is upright; false if shutdown was triggered.
    bool check_fall_down(float gravity_b_z) const
    {
        if (gravity_b_z > gravity_z_upper_) {
            on_shutdown_();
            throw std::runtime_error("Robot fell down! gravity_b.z="
                                     + std::to_string(gravity_b_z));
        }
        return true;
    }

private:
    float                       gravity_z_upper_{0.f};
    const std::vector<double>*  joint_limits_ = nullptr;   // non-owning pointer
    ShutdownCallback            on_shutdown_;
    bool                        configured_ = false;
};
