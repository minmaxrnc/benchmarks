# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import torch
from minmaxrnc import MinMaxRNC_LM as _MinMaxRNC_LM, MinMaxRNCLMConfig, MinMaxRNCConfig
from benchmarks.models.model import Model


class MinMaxRNC_LM(Model):
    """
    Benchmark wrapper around minmax.MinMaxRNC_LM.

    Constructor kwargs (all passed from the meta YAML):
        n_layers             – number of MinMaxLayers
        d_model              – residual-stream width
        d_state              – per-neuron hidden-state dimension
        head_dropout         – dropout before the LM head (default 0.0)
        tie_weights          – tie LM-head ↔ embedding weights (default True)
        default_unroll_steps – sequence chunk size; -1 → full sequence at once
                               Any value ≥ 1 limits peak memory at the cost
                               of slightly more sequential work.
        **backbone_kwargs    – any other MinMaxRNCConfig fields
    """

    def __init__(
        self,
        name:                 str,
        iosize:               int,
        n_layers:             int,
        d_model:              int,
        d_state:              int,
        head_dropout:         float = 0.0,
        tie_weights:          bool  = True,
        default_unroll_steps: int   = 64,
        s_r_init:             str   = 'small_init',
        **backbone_kwargs,
    ):
        super().__init__(name)
        self._iosize               = iosize

        self._backbone_cfg = MinMaxRNCConfig(
            d_model  = d_model,
            n_layers = n_layers,
            d_state  = d_state,
            s_r_init = s_r_init,
            **backbone_kwargs,
        )
        self._lm_cfg = MinMaxRNCLMConfig(
            backbone     = self._backbone_cfg,
            head_dropout = head_dropout,
            tie_weights  = tie_weights,
        )
        self._lm = _MinMaxRNC_LM(vocab_size=iosize, cfg=self._lm_cfg)


    def forward(
        self,
        x: torch.Tensor,
        state=None,
        unroll_steps: int = 1,
        return_state: bool = False
    ) -> torch.Tensor:
        logits, new_state = self._lm(x, unroll_steps, state=state, return_state=True)
        if return_state:
            return logits, new_state
        return logits

    def supports_unroll_steps(self) -> bool:
        return True

    def reset(self):
        self._lm = _MinMaxRNC_LM(vocab_size=self._iosize, cfg=self._lm_cfg)

