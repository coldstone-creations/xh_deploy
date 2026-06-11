#include "policy_manager.hpp"

#include <algorithm>
#include <stdexcept>

void PolicyManager::add_policy(PolicyConfig cfg) {
    const size_t idx = configs_.size();

    // Detect interrupt source
    const auto has_source = [&cfg](const std::string& name) {
        auto match = [&name](const ObsSourceSpec& s) { return s.name == name; };
        return std::any_of(cfg.obs_layout.begin(), cfg.obs_layout.end(), match) ||
               std::any_of(cfg.extra_obs_layout.begin(), cfg.extra_obs_layout.end(), match);
    };
    if (has_source("interrupt")) has_interrupt_ = true;

    // Detect motion
    cfg.has_motion = !cfg.motion_path.empty();
    if (cfg.has_motion) {
        motion_indices_.push_back(idx);
    }

    // Initialise runtime state
    PolicyRuntimeState st;
    st.obs.resize(cfg.obs_num, 0.0f);
    st.obs_segments.resize(cfg.obs_layout.size());
    for (size_t j = 0; j < cfg.obs_layout.size(); ++j) {
        st.obs_segments[j].resize(cfg.obs_layout[j].size, 0.0f);
    }
    st.extra_obs_segments.resize(cfg.extra_obs_layout.size());
    for (size_t j = 0; j < cfg.extra_obs_layout.size(); ++j) {
        st.extra_obs_segments[j].resize(cfg.extra_obs_layout[j].size, 0.0f);
    }

    configs_.push_back(std::move(cfg));
    states_.push_back(std::move(st));
    runners_.push_back(nullptr);   // ModelRunner is set up externally
    motions_.push_back(nullptr);   // MotionLoader is set up externally
}

ModelRunner& PolicyManager::active_runner() {
    return *runners_[active_idx_];
}

ModelRunner& PolicyManager::runner(size_t idx) {
    return *runners_[idx];
}

void PolicyManager::set_active(size_t idx) {
    if (idx >= configs_.size()) {
        throw std::out_of_range("PolicyManager::set_active: index out of range");
    }
    active_idx_ = idx;
}

void PolicyManager::reset_policy(size_t idx) {
    auto& st = states_[idx];
    std::fill(st.obs.begin(), st.obs.end(), 0.0f);
    for (auto& seg : st.obs_segments) std::fill(seg.begin(), seg.end(), 0.0f);
    for (auto& seg : st.extra_obs_segments) std::fill(seg.begin(), seg.end(), 0.0f);
    if (runners_[idx]) runners_[idx]->reset();
    st.motion_frame   = 0;
    st.is_first_frame = true;
}

void PolicyManager::reset_all() {
    for (size_t i = 0; i < configs_.size(); ++i) reset_policy(i);
}

void PolicyManager::step_motion_frame() {
    auto& st = states_[active_idx_];
    const auto& cfg = configs_[active_idx_];
    if (!cfg.has_motion) return;

    auto& loader = motions_[active_idx_];
    if (!loader) return;

    st.motion_frame += 1;
    if (st.motion_frame >= loader->get_num_frames()) {
        st.motion_frame = loader->get_num_frames() - 1;
    }
}

void PolicyManager::set_runner(size_t idx,
                               std::unique_ptr<ModelRunner> runner) {
    if (idx >= runners_.size()) {
        throw std::out_of_range("PolicyManager::set_runner: index out of range");
    }
    runners_[idx] = std::move(runner);
}

void PolicyManager::set_motion_loader(size_t idx,
                                      std::shared_ptr<MotionLoader> loader) {
    if (idx >= motions_.size()) {
        throw std::out_of_range("PolicyManager::set_motion_loader: index out of range");
    }
    motions_[idx] = std::move(loader);
}

std::shared_ptr<MotionLoader> PolicyManager::motion_loader(size_t idx) const {
    if (idx >= motions_.size()) return nullptr;
    return motions_[idx];
}
