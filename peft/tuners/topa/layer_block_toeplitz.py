# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class ToPALayer(BaseTunerLayer):
    adapter_layer_names = ("topa_toep_w",)
    other_param_names = ("topa_toep_r", "topa_block_size")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.topa_toep_r = {}
        self.topa_block_size = {}
        self.topa_toep_w = nn.ParameterDict({})
        self._disable_adapters = False
        self.merged_adapters = []

        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(
        self,
        adapter_name: str,
        toep_r: int,
        block_size: int,
        init_weights: bool,
        **kwargs,
    ) -> None:
        if toep_r < 0:
            raise ValueError(f"\`toep_r\` should be a positive integer value but the value passed is {toep_r}")

        if block_size <= 0:
            raise ValueError(f"\`block_size\` should be a positive integer value but the value passed is {block_size}")

        if self.in_features % block_size != 0:
            raise ValueError(f"\`in_features\` ({self.in_features}) must be divisible by \`block_size\` ({block_size})")

        self.topa_toep_r[adapter_name] = toep_r
        self.topa_block_size[adapter_name] = block_size

        num_blocks = self.in_features // block_size

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.topa_toep_w[adapter_name] = nn.Parameter(
                torch.zeros(num_blocks, num_blocks, 2 * block_size - 1, toep_r), 
                requires_grad=True
            )

        else:
            raise TypeError(f"ToPA is not implemented for base layers of type {type(base_layer).__name__}")

        self.reset_topa_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)


    def reset_topa_parameters(self, adapter_name: str):
        toep_r = self.topa_toep_r[adapter_name]

        with torch.no_grad():
            for i in range(toep_r):
                if i != 0:
                    self.topa_toep_w[adapter_name][:, :, :, i].normal_(mean=0.0, std=0.02)


    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.topa_toep_w.keys():
                continue

            warnings.warn("Scaling operation for block Toeplitz not supported! Automatically set scale to 1.")

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.topa_toep_w.keys():
                continue

            warnings.warn("Unscaling operation for block Toeplitz not supported! Keeping scale at 1.")


class ToPALinear(nn.Module, ToPALayer):

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        toep_r: int = 0,
        block_size: int = 128,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        ToPALayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, toep_r, block_size, init_weights, **kwargs)

    def block_toeplitz_multiplication(self, w_block, A_block, block_size, **kwargs):
        r = w_block[block_size - 1:]
        inv_c = w_block[0:block_size - 1]

        m = block_size
        n = block_size

        fft_len = int(2 ** math.ceil(math.log2(m + n - 1)))
        zp = fft_len - m - n + 1

        if A_block.shape[-1] != n:
            raise ValueError("A_block dimensions not compatible with Toeplitz block")

        dtype = torch.float32 if w_block.dtype == torch.bfloat16 else w_block.dtype

        x = torch.cat((r, torch.zeros(zp, dtype=dtype, device=r.device), inv_c))

        xf = torch.fft.rfft(x, n=fft_len)
        Af = torch.fft.rfft(A_block.to(dtype), n=fft_len, dim=-1)

        return torch.fft.irfft((Af * xf).T, n=fft_len, dim=0)[:n,].T.to(w_block.dtype)


    def apply_block_toeplitz(self, w, A, block_size, **kwargs):
        *batch_dims, in_features = A.shape
        num_blocks_row = in_features // block_size
        num_blocks_col = w.shape[1]

        A_blocks = A.reshape(*batch_dims, num_blocks_row, block_size)


        r = w[:, :, block_size - 1:]
        inv_c = w[:, :, 0:block_size - 1]

        m = block_size
        n = block_size
        fft_len = int(2 ** math.ceil(math.log2(m + n - 1)))
        zp = fft_len - m - n + 1

        dtype = torch.float32 if w.dtype == torch.bfloat16 else w.dtype

        zeros_pad = torch.zeros(num_blocks_row, num_blocks_col, zp, dtype=dtype, device=w.device)
        x = torch.cat((r, zeros_pad, inv_c), dim=-1)

        xf = torch.fft.rfft(x, n=fft_len, dim=-1)

        A_blocks_expanded = A_blocks.unsqueeze(-2)

        Af = torch.fft.rfft(A_blocks_expanded.to(dtype), n=fft_len, dim=-1)

        product = Af * xf.unsqueeze(0)


        product_transposed = product.permute(-1, *range(len(batch_dims)), -3, -2)

        ifft_result = torch.fft.irfft(product_transposed, n=fft_len, dim=0)

        ifft_result = ifft_result[:n, ...].permute(*range(1, len(batch_dims) + 1), -2, -1, 0)

        result_blocks = ifft_result.sum(dim=-3)

        out_features = num_blocks_col * block_size

        result = result_blocks.reshape(*batch_dims, out_features)

        return result


    def get_mult_delta_weight(self, adapter_name: str) -> torch.Tensor:
        toep_r = self.topa_toep_r[adapter_name]
        block_size = self.topa_block_size[adapter_name]
        device = self.topa_toep_w[adapter_name].device
        dtype = self.topa_toep_w[adapter_name].dtype

        x = torch.eye(self.in_features, dtype=dtype).to(device)

        weight = self.get_multed_output(adapter_name, x)

        return weight


    def get_multed_output(self, adapter_name: str, x: torch.Tensor) -> torch.Tensor:


        toep_r = self.topa_toep_r[adapter_name]
        block_size = self.topa_block_size[adapter_name]

        for i in range(toep_r - 1, -1, -1):
            w = self.topa_toep_w[adapter_name][:, :, :, i]

            x = self.apply_block_toeplitz(w, x, block_size)

        return x


    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.topa_toep_w.keys():
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.weight.dtype
                if safe_merge:
                    orig_weight = base_layer.weight.data.clone()
                    multed_delta_weight = self.get_mult_delta_weight(active_adapter)


                    new_weight = torch.mm(multed_delta_weight, orig_weight.T.to(multed_delta_weight.dtype)).T + orig_weight.to(multed_delta_weight.dtype)

                    if not torch.isfinite(new_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = new_weight.to(orig_dtype)
                else:

                    multed_delta_weight = self.get_mult_delta_weight(active_adapter)


                    new_weight = torch.mm(multed_delta_weight, base_layer.weight.data.T.to(multed_delta_weight.dtype)).T + base_layer.weight.data.to(multed_delta_weight.dtype)

                    base_layer.weight.data = new_weight.to(orig_dtype)

                self.merged_adapters.append(active_adapter)


    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        raise NotImplementedError("ToPA unmerge is not implemented")


    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)

        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)

        else:
            orig_weight = self.get_base_layer().weight.data

            def calculate_mult_output(x):

                for active_adapter in self.active_adapters:
                    if active_adapter not in self.topa_toep_w.keys():
                        continue

                    x_transformerd = self.get_multed_output(active_adapter, x)

                x_transformerd = x_transformerd.to(orig_weight.dtype)

                x = x.to(orig_weight.dtype)

                result = F.linear(input=x_transformerd + x, weight=orig_weight, bias=self.get_base_layer().bias)

                return result

            result = calculate_mult_output(x)

        result = result.to(previous_dtype)
        return result


    def __repr__(self) -> str:
        rep = super().__repr__()
        return "topa." + rep
