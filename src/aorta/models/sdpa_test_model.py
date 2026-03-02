"""SDPA Test Model for integration with AORTA training infrastructure.

This module provides a minimal nn.Module wrapper around SDPA backward operations
to enable testing through the existing train.py and fsdp_trainer.py infrastructure.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

log = logging.getLogger(__name__)


@dataclass
class SDPATestConfig:
    """Configuration for SDPA test model."""
    
    input_dir: str = "/home/vivekag/scratch/apps/aorta_work/nan_issue/sdpa/input"
    """Directory containing SDPA input files"""
    
    iteration_counter: int = 0
    """Internal counter for tracking iterations"""
    
    max_iterations: int = 1000
    """Maximum number of iterations before stopping"""
    
    verbose: bool = False
    """Enable verbose logging"""
    
    check_nan: bool = True
    """Check for NaN/Inf in outputs"""
    
    # Dummy model parameters for compatibility
    vocab_size: int = 100
    embedding_dim: int = 64
    num_dense_features: int = 1
    dense_dim: int = 64
    model_dim: int = 64
    num_heads: int = 1
    num_layers: int = 1
    dropout: float = 0.0
    mlp_hidden_dim: int = 64


def load_tensor(file_path: str) -> Optional[torch.Tensor]:
    """Load a tensor from a local file."""
    if not os.path.exists(file_path):
        return None
    try:
        tensor = torch.load(file_path, map_location="cpu", weights_only=False)
        return tensor
    except Exception as e:
        log.warning(f"Could not load {file_path}: {e}")
        return None


def load_metadata(file_path: str) -> Optional[dict]:
    """Load metadata JSON from a local file."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not load metadata from {file_path}: {e}")
        return None


class SDPATestModel(nn.Module):
    """
    Minimal model wrapper for SDPA backward testing.
    
    This model wraps SDPA backward operations in an nn.Module interface
    for compatibility with AORTA's training infrastructure. It can be
    wrapped with FSDP and used in distributed training loops.
    
    The model:
    1. Loads SDPA inputs once during initialization
    2. Runs SDPA backward on each forward pass
    3. Returns a dummy loss for training loop compatibility
    4. Checks for NaN/Inf in outputs if enabled
    """
    
    def __init__(self, cfg: SDPATestConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.iteration = 0
        
        # Create a dummy parameter to make FSDP happy
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
        # Load SDPA inputs
        self.inputs = self._load_inputs()
        
        if not self.inputs:
            log.warning("Failed to load SDPA inputs. Model will not function correctly.")
        else:
            log.info(f"SDPATestModel initialized with inputs from {cfg.input_dir}")
            log.info(f"Max iterations: {cfg.max_iterations}")
    
    def _load_inputs(self) -> Dict[str, Any]:
        """Load SDPA inputs from saved files."""
        inputs = {}
        
        log.info(f"Loading SDPA inputs from {self.cfg.input_dir}")
        
        # Load metadata
        metadata_path = os.path.join(self.cfg.input_dir, "metadata.json")
        metadata = load_metadata(metadata_path)
        if not metadata:
            log.error(f"ERROR: metadata.json not found in {self.cfg.input_dir}")
            return inputs
        
        log.info(f"Loaded metadata: func={metadata.get('func_name')}, rank={metadata.get('rank')}")
        
        saved_inputs = metadata.get("saved_inputs", {})
        
        # Load tensor inputs
        tensor_names = [
            "grad_out",
            "query",
            "key",
            "value",
            "out",
            "logsumexp",
            "cum_seq_q",
            "cum_seq_k",
            "philox_seed",
            "philox_offset",
        ]
        
        for name in tensor_names:
            value = saved_inputs.get(name, "None")
            if value in ["None", "null", None]:
                continue
            
            file_path = os.path.join(self.cfg.input_dir, value)
            tensor = load_tensor(file_path)
            if tensor is not None:
                inputs[name] = tensor
                if self.cfg.verbose:
                    log.info(f"Loaded: {name} - shape={list(tensor.shape)}, dtype={tensor.dtype}")
        
        # Load scalar inputs
        scalar_names = ["max_q", "max_k", "dropout_p", "is_causal", "scale"]
        
        for name in scalar_names:
            value = saved_inputs.get(name, "None")
            if value in ["None", "null", None]:
                continue
            try:
                if name in ["max_q", "max_k"]:
                    inputs[name] = int(value)
                elif name in ["dropout_p", "scale"]:
                    inputs[name] = float(value)
                elif name == "is_causal":
                    inputs[name] = str(value).lower() in ["true", "1"]
                if self.cfg.verbose:
                    log.info(f"Loaded scalar: {name} = {inputs[name]}")
            except (ValueError, AttributeError) as e:
                log.warning(f"Could not parse {name}={value}: {e}")
        
        # Validate required inputs
        required_inputs = ["grad_out", "query", "key", "value", "out", "logsumexp"]
        missing_inputs = [name for name in required_inputs if name not in inputs]
        if missing_inputs:
            log.error(f"ERROR: Missing required inputs: {missing_inputs}")
            return {}
        
        return inputs
    
    def _check_nan_inf(
        self, tensors: Tuple[torch.Tensor, ...], tensor_names: list[str]
    ) -> bool:
        """Check for NaN/Inf in tensors. Returns True if found."""
        found_issue = False
        
        for name, tensor in zip(tensor_names, tensors):
            nan_mask = torch.isnan(tensor)
            has_nan = bool(nan_mask.any().item())
            nan_count = int(nan_mask.sum().item())
            
            inf_mask = torch.isinf(tensor)
            has_inf = bool(inf_mask.any().item())
            inf_count = int(inf_mask.sum().item())
            
            if has_nan or has_inf:
                found_issue = True
                log.warning(
                    f"Iteration {self.iteration}: NaN/Inf in {name} - "
                    f"NaN={has_nan} (count={nan_count}), Inf={has_inf} (count={inf_count})"
                )
        
        return found_issue
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run SDPA backward operation and return dummy loss.
        
        Args:
            batch: Input batch (ignored, we use pre-loaded SDPA inputs)
        
        Returns:
            Dummy loss tensor for training loop compatibility
        """
        self.iteration += 1
        
        if not self.inputs:
            log.error(f"Iteration {self.iteration}: No inputs loaded, returning zero loss")
            return torch.tensor(0.0, device=self.dummy_param.device)
        
        device = self.dummy_param.device
        
        # Move inputs to device
        grad_out = self.inputs["grad_out"].to(device)
        query = self.inputs["query"].to(device)
        key = self.inputs["key"].to(device)
        value = self.inputs["value"].to(device)
        out = self.inputs["out"].to(device)
        logsumexp = self.inputs["logsumexp"].to(device)
        
        cum_seq_q = self.inputs.get("cum_seq_q")
        if cum_seq_q is not None:
            cum_seq_q = cum_seq_q.to(device)
        
        cum_seq_k = self.inputs.get("cum_seq_k")
        if cum_seq_k is not None:
            cum_seq_k = cum_seq_k.to(device)
        
        max_q = self.inputs.get("max_q", 0)
        max_k = self.inputs.get("max_k", 0)
        dropout_p = self.inputs.get("dropout_p", 0.0)
        is_causal = self.inputs.get("is_causal", False)
        
        philox_seed = self.inputs.get("philox_seed")
        if philox_seed is not None:
            philox_seed = philox_seed.to(device)
        
        philox_offset = self.inputs.get("philox_offset")
        if philox_offset is not None:
            philox_offset = philox_offset.to(device)
        
        scale = self.inputs.get("scale")
        
        # Run SDPA backward
        if self.cfg.verbose and self.iteration % 10 == 0:
            log.info(f"Iteration {self.iteration}/{self.cfg.max_iterations}: Running SDPA backward...")
        
        try:
            result = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
                grad_out,
                query,
                key,
                value,
                out,
                logsumexp,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                dropout_p,
                is_causal,
                philox_seed,
                philox_offset,
                scale=scale,
            )
            
            grad_query = result[0]
            grad_key = result[1]
            grad_value = result[2]
            
            # Check for NaN/Inf
            if self.cfg.check_nan:
                found_nan = self._check_nan_inf(
                    (grad_query.detach().cpu(), grad_key.detach().cpu(), grad_value.detach().cpu()),
                    ["grad_query", "grad_key", "grad_value"],
                )
                
                if found_nan:
                    log.error(f"Iteration {self.iteration}: NaN/Inf DETECTED!")
                    # Return a loss that will be detected as NaN
                    return torch.tensor(float('nan'), device=device)
            
            # Return dummy loss (sum of gradient norms for backward compatibility)
            loss = grad_query.norm() + grad_key.norm() + grad_value.norm()
            return loss
        
        except Exception as e:
            log.error(f"Iteration {self.iteration}: ERROR running SDPA backward: {e}")
            # Return NaN loss to signal failure
            return torch.tensor(float('nan'), device=device)


__all__ = ["SDPATestConfig", "SDPATestModel"]
