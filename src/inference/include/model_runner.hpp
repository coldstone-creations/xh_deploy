#pragma once

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

/// @brief Owns one ONNX Runtime session and its input / output tensors.
///
/// One ModelRunner per policy.  Non-copyable; move-only if needed (though
/// InferenceNode stores them in a vector of unique_ptr).
class ModelRunner {
public:
    /// Construct with references to the shared ONNX environment and allocator.
    /// Call setup_model() afterwards to load a specific model file.
    ModelRunner(Ort::Env& env,
                Ort::AllocatorWithDefaultOptions& allocator,
                int joint_num)
        : env_(env)
        , allocator_(allocator)
        , joint_num_(joint_num)
    {}

    /// Load an ONNX model and allocate tensors.
    /// @param model_path  Absolute path to the .onnx file.
    /// @param input_size  Total number of float inputs expected.
    /// @throws std::runtime_error on shape mismatch or unsupported config.
    void setup_model(const std::string& model_path, int input_size);

    /// Execute one inference step.
    /// Reads from input_buffer(), writes to output_buffer().
    void run();

    /// Mutable access to the flattened input buffer (pre-allocated).
    std::vector<float>& input_buffer() { return input_buffer_; }

    /// Read-only access to the output buffer.
    const std::vector<float>& output_buffer() const { return output_buffer_; }

    /// Read-only access to the output buffer (mutable for in-place clamping).
    std::vector<float>& output_buffer() { return output_buffer_; }

    /// Zero out both input and output buffers.
    void reset();

private:
    Ort::Env&                             env_;
    Ort::AllocatorWithDefaultOptions&     allocator_;
    int                                   joint_num_;

    std::unique_ptr<Ort::Session>         session_;
    std::unique_ptr<Ort::MemoryInfo>      memory_info_;
    std::unique_ptr<Ort::Value>           input_tensor_;
    std::unique_ptr<Ort::Value>           output_tensor_;

    std::vector<std::string>              input_names_;
    std::vector<std::string>              output_names_;
    std::vector<const char*>              input_names_raw_;
    std::vector<const char*>              output_names_raw_;
    std::vector<int64_t>                  input_shape_;
    std::vector<int64_t>                  output_shape_;

    std::vector<float>                    input_buffer_;
    std::vector<float>                    output_buffer_;
};
