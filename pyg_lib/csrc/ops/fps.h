#pragma once

#include <ATen/ATen.h>
#include <torch/script.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

// Performs iterative farthest point sampling of src points
PYG_API torch::Tensor fps(torch::Tensor src, torch::Tensor ptr, double ratio,
                  bool random_start);

}  // namespace ops
}  // namespace pyg
