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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class ToPAConfig(PeftConfig):

    toep_r: int = field(
        default=4,
        metadata={
            "help": "The rank of Toeplitz component across different layers.",
        },
    )
    block_size: int = field(
        default=128,
        metadata={
            "help": "The block size for block Toeplitz matrices. Input features must be divisible by this value.",
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with ToPA layers.",
            "example": "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' ",
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude."},
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern. "
            "This should target the `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`."
        },
    )
    bias: str = field(default="none", metadata={"help": "Bias type. Can be 'none', 'all' or 'topa_only'"})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.TOPA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")

        if self.block_size <= 0:
            raise ValueError(f"`block_size` must be a positive integer, got {self.block_size}")
