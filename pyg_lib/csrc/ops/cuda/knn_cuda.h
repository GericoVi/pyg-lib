#pragma once

#include "pyg_lib/csrc/macros.h"
#include <torch/torch.h>

torch::Tensor knn_cuda(torch::Tensor x, torch::Tensor y,
                       torch::optional<torch::Tensor> ptr_x,
                       torch::optional<torch::Tensor> ptr_y, int64_t k,
                       bool cosine);
