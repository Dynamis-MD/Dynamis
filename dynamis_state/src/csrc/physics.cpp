#include "dynamis_state/physics.hpp"

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

} // namespace dynamis_state
