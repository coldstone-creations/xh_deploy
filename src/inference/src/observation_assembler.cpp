#include "observation_assembler.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string trim_copy(const std::string& value) {
    const auto first = std::find_if_not(value.begin(), value.end(),
                                        [](unsigned char c) { return std::isspace(c) != 0; });
    const auto last  = std::find_if_not(value.rbegin(), value.rend(),
                                        [](unsigned char c) { return std::isspace(c) != 0; }).base();
    if (first >= last) return "";
    return std::string(first, last);
}

std::vector<std::string> split_obs_layout_spec(const std::string& layout_spec) {
    std::vector<std::string> tokens;
    size_t start = 0;
    while (start < layout_spec.size()) {
        const size_t end = layout_spec.find(',', start);
        const std::string token = trim_copy(
            layout_spec.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (!token.empty()) tokens.push_back(token);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return tokens;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// parse_layout
// ---------------------------------------------------------------------------
std::vector<ObsSourceSpec> ObservationAssembler::parse_layout(
    const std::string& layout_spec,
    const std::string& layout_name)
{
    const std::vector<std::string> specs = split_obs_layout_spec(layout_spec);
    if (specs.empty()) {
        throw std::runtime_error(layout_name + " must be explicitly configured");
    }

    std::vector<ObsSourceSpec> layout;
    layout.reserve(specs.size());

    for (const std::string& raw_spec : specs) {
        const std::string spec = trim_copy(raw_spec);
        const size_t separator = spec.find(':');
        if (separator == std::string::npos || separator == 0 || separator == spec.size() - 1) {
            throw std::runtime_error(layout_name + " entry must use 'name:size' format: " + raw_spec);
        }

        const std::string name      = trim_copy(spec.substr(0, separator));
        const std::string size_text = trim_copy(spec.substr(separator + 1));

        if (name.empty() || size_text.empty()) {
            throw std::runtime_error(layout_name + " entry must use 'name:size' format: " + raw_spec);
        }
        if (!std::all_of(size_text.begin(), size_text.end(),
                         [](unsigned char c) { return std::isdigit(c) != 0; })) {
            throw std::runtime_error(layout_name + " field size must be a positive integer: " + raw_spec);
        }

        ObsSourceSpec src;
        src.name = name;
        src.size = std::stoi(size_text);
        layout.push_back(src);
    }
    return layout;
}

// ---------------------------------------------------------------------------
// assemble
// ---------------------------------------------------------------------------
void ObservationAssembler::assemble(
    std::vector<std::vector<float>>& segments,
    const std::vector<ObsSourceSpec>& layout) const
{
    for (size_t i = 0; i < layout.size(); ++i) {
        auto it = fillers_.find(layout[i].name);
        if (it == fillers_.end()) {
            throw std::runtime_error("Filler not registered: " + layout[i].name);
        }
        it->second(segments[i]);
    }
}

// ---------------------------------------------------------------------------
// flatten
// ---------------------------------------------------------------------------
void ObservationAssembler::flatten(
    const std::vector<std::vector<float>>& segments,
    std::vector<float>::iterator output_begin)
{
    int offset = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
        std::copy(segments[i].begin(), segments[i].end(), output_begin + offset);
        offset += static_cast<int>(segments[i].size());
    }
}

// ---------------------------------------------------------------------------
// stack_frames
// ---------------------------------------------------------------------------
void ObservationAssembler::stack_frames(
    std::vector<float>& input_buffer,
    const std::vector<float>& obs,
    int obs_num,
    int frame_stack,
    ObsStackOrder stack_order,
    const std::vector<int>& field_sizes,
    bool is_first_frame)
{
    if (stack_order == ObsStackOrder::FrameMajor) {
        if (is_first_frame) {
            for (int frame = 0; frame < frame_stack; ++frame) {
                std::copy(obs.begin(), obs.end(),
                          input_buffer.begin() + frame * obs_num);
            }
        } else {
            std::move(input_buffer.begin() + obs_num,
                      input_buffer.begin() + frame_stack * obs_num,
                      input_buffer.begin());
            std::copy(obs.begin(), obs.end(),
                      input_buffer.begin() + (frame_stack - 1) * obs_num);
        }
        return;
    }

    // ObsMajor
    int input_offset = 0;
    int obs_offset   = 0;
    for (const int field_size : field_sizes) {
        if (is_first_frame) {
            for (int frame = 0; frame < frame_stack; ++frame) {
                std::copy(obs.begin() + obs_offset,
                          obs.begin() + obs_offset + field_size,
                          input_buffer.begin() + input_offset + frame * field_size);
            }
        } else {
            std::move(input_buffer.begin() + input_offset + field_size,
                      input_buffer.begin() + input_offset + frame_stack * field_size,
                      input_buffer.begin() + input_offset);
            std::copy(obs.begin() + obs_offset,
                      obs.begin() + obs_offset + field_size,
                      input_buffer.begin() + input_offset + (frame_stack - 1) * field_size);
        }
        input_offset += field_size * frame_stack;
        obs_offset   += field_size;
    }
}

// ---------------------------------------------------------------------------
// register_standard_fillers
// ---------------------------------------------------------------------------
void ObservationAssembler::register_standard_fillers(const StandardFillers& f) {
    if (f.cmd_vel)     register_filler("cmd_vel",     f.cmd_vel);
    if (f.ang_vel)     register_filler("ang_vel",     f.ang_vel);
    if (f.gravity_b)   register_filler("gravity_b",   f.gravity_b);
    if (f.dof_pos)     register_filler("dof_pos",     f.dof_pos);
    if (f.dof_vel)     register_filler("dof_vel",     f.dof_vel);
    if (f.last_action) register_filler("last_action", f.last_action);
    if (f.interrupt)   register_filler("interrupt",   f.interrupt);
    if (f.perception)  register_filler("perception",  f.perception);
    if (f.motion_pos)  register_filler("motion_pos",  f.motion_pos);
    if (f.motion_vel)  register_filler("motion_vel",  f.motion_vel);
}
