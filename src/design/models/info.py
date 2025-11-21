#!/usr/bin/python
# -*- coding:utf-8 -*-
from dataclasses import dataclass

import torch


@dataclass
class ComplexInfo:
    chain_ids: list
    chain_orders: dict   # chain id to chain order
    generate_mask: torch.Tensor