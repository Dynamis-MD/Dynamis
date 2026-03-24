#pragma once
#include <torch/extension.h>

namespace distnnp_model
{

at::Tensor push_buffer_create(at::Tensor input, at::Tensor buffer, at::Tensor input_locs);

} // namespace distnnp_model
