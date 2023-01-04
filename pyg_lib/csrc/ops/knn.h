#pragma once

#include <ATen/ATen.h>
#include <torch/script.h>
#include "pyg_lib/csrc/macros.h"

namespace pyg {
namespace ops {

// Generates a kNN graph
PYG_API torch::Tensor knn(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                  torch::Tensor ptr_y, int64_t k, bool cosine);

}  // namespace ops
}  // namespace pyg
