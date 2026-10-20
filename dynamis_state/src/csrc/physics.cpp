#include "dynamis_state/physics.hpp"
#include <ATen/CPUGeneratorImpl.h>
#include <stdexcept>
#include <string>

namespace dynamis_state {

torch::Tensor mass_table()
{
    torch::Tensor table = torch::zeros({ 119 }, torch::kFloat32);
    auto acc = table.accessor<float, 1>();
    for (const auto &[label, element] : periodic_table) {
        acc[element.number] = element.mass;
    }
    return table;
}

torch::Tensor sample_maxwell_boltzmann(const torch::Tensor &atom_types, double temperature_K,
                                       uint64_t seed)
{
    // negated so NaN is rejected too
    if (!(temperature_K >= 0.0)) {
        throw std::invalid_argument("temperature_K must be non-negative, got " +
                                    std::to_string(temperature_K));
    }

    torch::Tensor types = atom_types.reshape(-1).to(torch::kCPU, torch::kLong);
    torch::Tensor table = mass_table().to(torch::kFloat64);
    torch::Tensor known = (types > 0) & (types < table.size(0));
    if (!known.all().item<bool>()) {
        int64_t i = (~known).nonzero()[0].item<int64_t>();
        throw std::invalid_argument("atom " + std::to_string(i) + " has unknown atom type " +
                                    std::to_string(types[i].item<int64_t>()) +
                                    ", cannot sample its velocity");
    }
    torch::Tensor masses = table.index_select(0, types);

    // a dedicated generator leaves the global torch RNG untouched
    auto generator = at::make_generator<at::CPUGeneratorImpl>(seed);
    torch::Tensor velocities = torch::randn({ types.size(0), 3 }, generator, torch::kFloat64);
    velocities *= torch::sqrt(kBoltzmann * temperature_K / masses).unsqueeze(1);

    // remove centre-of-mass drift so the system doesn't translate as a whole
    if (types.size(0) > 0) {
        velocities -= (masses.unsqueeze(1) * velocities).sum(0) / masses.sum();
    }
    return velocities.to(atom_types.device(), torch::kFloat32);
}

} // namespace dynamis_state
