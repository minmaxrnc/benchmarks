# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

from typing import Optional, Sequence

import torch
from minmaxrnc import (
    MinMaxRNC_LM as _MinMaxRNC_LM,
    MinMaxRNCLMConfig,
    MinMaxRNCConfig,
    NeuronArray,
)
from benchmarks.models.model import Model


class MinMaxRNC_LM(Model):
    """
    Benchmark wrapper around minmaxrnc.MinMaxRNC_LM (minmaxrnc >= 0.4.1).

    Constructor kwargs (all passed from the meta YAML):
        n_layers             – number of MinMaxLayers
        d_model              – residual-stream width
        scalar_neurons       – number of scalar-state neurons per layer
        d_state              – legacy name of scalar_neurons (set at most one
                               of the two); d_state=N builds the same model as
                               scalar_neurons=N
        neurons              – further neuron arrays per layer, as a list of
                               [d_state, num_neurons] pairs, e.g. [[3, 16]]
                               for 16 neurons of state dimension 3
        amp_dtype            – model-level autocast: None (default, fp32),
                               'bf16' or 'fp16'. With AMP on, logits are cast
                               back to float32 so that the loss is computed in
                               full precision
        recurrence_dtype     – precision of the recurrence: None (default),
                               'fp32', 'bf16' or 'fp16'
        head_dropout         – dropout before the LM head (default 0.0)
        tie_weights          – tie LM-head ↔ embedding weights (default True)
        output_gate          – gate the neuron outputs (MinMaxRNCLMConfig)
        conv_type            – 'basic' or 'gated' (MinMaxRNCLMConfig)
        ffn_dropout          – dropout inside the FFNs (MinMaxRNCLMConfig)
        s_r_init             – 'small_init', 'kaiming' or 'asymmetric'
        default_unroll_steps – unused, kept for compatibility with old configs
        **backbone_kwargs    – any other MinMaxRNCConfig fields
    """

    def __init__(
        self,
        name:                 str,
        iosize:               int,
        n_layers:             int,
        d_model:              int,
        d_state:              Optional[int] = None,
        head_dropout:         float = 0.0,
        tie_weights:          bool  = True,
        default_unroll_steps: int   = 64,
        s_r_init:             str   = 'small_init',
        output_gate:          bool  = True,
        conv_type:            str   = 'basic',
        ffn_dropout:          float = 0.1,
        scalar_neurons:       Optional[int] = None,
        neurons:              Optional[Sequence] = None,
        amp_dtype:            Optional[str] = None,
        recurrence_dtype:     Optional[str] = None,
        **backbone_kwargs,
    ):
        super().__init__(name)
        self._iosize    = iosize
        self._amp_dtype = amp_dtype

        if d_state is not None and scalar_neurons is not None:
            raise ValueError(
                f"Set either scalar_neurons or its legacy name d_state, not both "
                f"(got scalar_neurons={scalar_neurons}, d_state={d_state})."
            )
        if scalar_neurons is None:
            scalar_neurons = d_state if d_state is not None else 0

        self._backbone_cfg = MinMaxRNCConfig(
            d_model          = d_model,
            n_layers         = n_layers,
            scalar_neurons   = scalar_neurons,
            neurons          = self._neuron_arrays(neurons),
            s_r_init         = s_r_init,
            amp_dtype        = amp_dtype,
            recurrence_dtype = recurrence_dtype,
            **backbone_kwargs,
        )
        self._lm_cfg = MinMaxRNCLMConfig(
            backbone     = self._backbone_cfg,
            head_dropout = head_dropout,
            tie_weights  = tie_weights,
            output_gate  = output_gate,
            conv_type    = conv_type,
            ffn_dropout  = ffn_dropout,
        )
        self._lm = _MinMaxRNC_LM(vocab_size=iosize, cfg=self._lm_cfg)


    @staticmethod
    def _neuron_arrays(neurons):
        """[[d_state, num_neurons], ...] (as written in the meta YAML) -> tuple of NeuronArray."""
        if neurons is None:
            return ()
        return tuple(NeuronArray.coerce(tuple(n) if isinstance(n, list) else n) for n in neurons)


    def forward(
        self,
        x: torch.Tensor,
        state=None,
        unroll_steps: int = 1,
        return_state: bool = False
    ) -> torch.Tensor:
        logits, new_state = self._lm(x, unroll_steps, state=state, return_state=True)
        if self._amp_dtype is not None:
            logits = logits.float()
        if return_state:
            return logits, new_state
        return logits

    def supports_unroll_steps(self) -> bool:
        return True

    def inductor_config(self) -> dict:
        """minmaxrnc 0.4 known issue: Inductor fails for neuron arrays of
        state dimension d > 1 in fp32 inference unless tiling-coalescing
        analysis is off (no measured change in speed); see the minmaxrnc
        CHANGELOG, 0.4.0, Known issues."""
        if any(array.d_state > 1 for array in self._backbone_cfg.neurons):
            return {'triton.coalesce_tiling_analysis': False}
        return {}


    def reset(self):
        self._lm = _MinMaxRNC_LM(vocab_size=self._iosize, cfg=self._lm_cfg)
