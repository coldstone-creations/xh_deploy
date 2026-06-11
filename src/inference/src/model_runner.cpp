#include "model_runner.hpp"

void ModelRunner::setup_model(const std::string& model_path, int input_size) {
    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(),
                                              session_options);

    const size_t num_inputs = session_->GetInputCount();
    if (num_inputs != 1) {
        throw std::runtime_error(
            "Only single-input ONNX models are supported: " + model_path);
    }
    input_names_.resize(num_inputs);

    for (size_t i = 0; i < num_inputs; ++i) {
        Ort::AllocatedStringPtr name =
            session_->GetInputNameAllocated(i, allocator_);
        input_names_[i] = name.get();
        auto type_info = session_->GetInputTypeInfo(i);
        input_shape_ = type_info.GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape_[0] == -1) input_shape_[0] = 1;
    }

    size_t model_input_size = 1;
    for (size_t i = 0; i < input_shape_.size(); ++i) {
        model_input_size *= static_cast<size_t>(input_shape_[i]);
    }
    if (model_input_size != static_cast<size_t>(input_size)) {
        throw std::runtime_error(
            "ONNX input size mismatch for " + model_path +
            ": model expects " + std::to_string(model_input_size) +
            " values, but config provides " + std::to_string(input_size));
    }
    input_buffer_.resize(input_size);

    const size_t num_outputs = session_->GetOutputCount();
    output_names_.resize(num_outputs);
    output_buffer_.resize(joint_num_);

    for (size_t i = 0; i < num_outputs; ++i) {
        Ort::AllocatedStringPtr name =
            session_->GetOutputNameAllocated(i, allocator_);
        output_names_[i] = name.get();
        auto type_info = session_->GetOutputTypeInfo(i);
        output_shape_ = type_info.GetTensorTypeAndShapeInfo().GetShape();
    }

    input_names_raw_.resize(num_inputs, nullptr);
    output_names_raw_.resize(num_outputs, nullptr);
    for (size_t i = 0; i < num_inputs; ++i)
        input_names_raw_[i] = input_names_[i].c_str();
    for (size_t i = 0; i < num_outputs; ++i)
        output_names_raw_[i] = output_names_[i].c_str();

    memory_info_ = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));

    input_tensor_ = std::make_unique<Ort::Value>(
        Ort::Value::CreateTensor<float>(*memory_info_, input_buffer_.data(),
                                        input_buffer_.size(),
                                        input_shape_.data(),
                                        input_shape_.size()));

    output_tensor_ = std::make_unique<Ort::Value>(
        Ort::Value::CreateTensor<float>(*memory_info_, output_buffer_.data(),
                                        output_buffer_.size(),
                                        output_shape_.data(),
                                        output_shape_.size()));
}

void ModelRunner::run() {
    session_->Run(Ort::RunOptions{nullptr},
                  input_names_raw_.data(),  input_tensor_.get(),
                  static_cast<size_t>(input_names_raw_.size()),
                  output_names_raw_.data(), output_tensor_.get(),
                  static_cast<size_t>(output_names_raw_.size()));
}

void ModelRunner::reset() {
    std::fill(input_buffer_.begin(),  input_buffer_.end(),  0.0f);
    std::fill(output_buffer_.begin(), output_buffer_.end(), 0.0f);
}
