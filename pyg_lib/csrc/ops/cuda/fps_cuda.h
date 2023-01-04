#pragma once

#include "pyg_lib/csrc/macros.h"
#include <torch/torch.h>

torch::Tensor fps_cuda(torch::Tensor src, torch::Tensor ptr,
                       torch::Tensor ratio, bool random_start);
