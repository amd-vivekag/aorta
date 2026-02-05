"""
Manual training correctness verification module.

This module provides utilities to detect silent corruption from race conditions
by comparing training outputs against known-correct reference values.

Based on academic research:
- Bottou et al. (2018): Deterministic comparison methodology
- You et al. (2020): Silent corruption detection in BERT training
- Hoffer et al. (2017): Variance-based corruption detection
- Chen et al. (2016): Gradient norm outlier detection

Key insight: Most torn reads produce valid but WRONG values (silent corruption),
not NaN. These verification methods detect corruption that doesn't produce NaN.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


log = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result from a correctness verification check."""

    method: str
    """Verification method name (e.g., 'deterministic', 'variance', 'gradient_norm')."""

    passed: bool
    """Whether the verification passed (no corruption detected)."""

    details: Dict[str, Any] = field(default_factory=dict)
    """Detailed metrics from the verification."""

    message: str = ""
    """Human-readable summary message."""


def set_deterministic_mode(seed: int = 42) -> None:
    """
    Enable fully deterministic mode for comparison.

    Reference: Paszke et al. (2019) - PyTorch deterministic mode guarantees.

    Args:
        seed: Random seed for all RNGs.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Use deterministic algorithms (may be slower but guarantees reproducibility)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        log.warning("torch.use_deterministic_algorithms not available; some ops may be non-deterministic")


# =============================================================================
# Method 1: Deterministic Comparison (Gold Standard)
# =============================================================================


def run_reference_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    loss_fn: Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
    autocast_dtype: Optional[torch.dtype] = None,
    with_sync: bool = True,
) -> Dict[str, Any]:
    """
    Run a single training step with full synchronization (correct baseline).

    This captures the "ground truth" values that should match if there's no race.

    Args:
        model: The model to train.
        optimizer: The optimizer.
        batch: Input batch (already on GPU).
        loss_fn: Loss function (scores, batch) -> loss.
        autocast_dtype: Mixed precision dtype (None for fp32).
        with_sync: If True, synchronize CUDA before and after (ensures H2D complete).

    Returns:
        Dict with loss, gradient norms, and parameter checksums.
    """
    if with_sync:
        torch.cuda.synchronize()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    if autocast_dtype:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            scores = model(batch)
            loss = loss_fn(scores, batch)
    else:
        scores = model(batch)
        loss = loss_fn(scores, batch)

    loss.backward()
    optimizer.step()

    if with_sync:
        torch.cuda.synchronize()

    # Capture reference values
    result = {
        "loss": loss.detach().cpu().item(),
        "gradient_norms": {},
        "parameter_checksums": {},
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            result["gradient_norms"][name] = param.grad.norm(2).item()
        result["parameter_checksums"][name] = param.data.sum().item()

    return result


def verify_deterministic_comparison(
    reference: Dict[str, Any],
    test: Dict[str, Any],
    loss_tolerance: float = 1e-5,
    grad_tolerance: float = 1e-4,
) -> VerificationResult:
    """
    Compare test run against reference for corruption detection.

    Reference: Bottou et al. (2018), You et al. (2020).

    Args:
        reference: Results from run_reference_step with sync.
        test: Results from run_reference_step without sync (may have race).
        loss_tolerance: Maximum allowed loss difference.
        grad_tolerance: Maximum allowed gradient norm difference.

    Returns:
        VerificationResult indicating if corruption was detected.
    """
    loss_diff = abs(reference["loss"] - test["loss"])
    loss_passed = loss_diff <= loss_tolerance

    # Check gradient norms
    grad_diffs = {}
    max_grad_diff = 0.0
    for name in reference["gradient_norms"]:
        ref_norm = reference["gradient_norms"].get(name, 0.0)
        test_norm = test["gradient_norms"].get(name, 0.0)
        diff = abs(ref_norm - test_norm)
        grad_diffs[name] = diff
        max_grad_diff = max(max_grad_diff, diff)

    grad_passed = max_grad_diff <= grad_tolerance

    passed = loss_passed and grad_passed

    if passed:
        message = f"PASSED: Loss diff={loss_diff:.2e}, max grad diff={max_grad_diff:.2e}"
    else:
        message = (
            f"FAILED: Loss diff={loss_diff:.2e} (tol={loss_tolerance:.2e}), "
            f"max grad diff={max_grad_diff:.2e} (tol={grad_tolerance:.2e}) - "
            f"RACE DETECTED! Training is silently corrupted."
        )

    return VerificationResult(
        method="deterministic_comparison",
        passed=passed,
        details={
            "loss_diff": loss_diff,
            "loss_tolerance": loss_tolerance,
            "max_grad_diff": max_grad_diff,
            "grad_tolerance": grad_tolerance,
            "grad_diffs": grad_diffs,
        },
        message=message,
    )


# =============================================================================
# Method 2: Statistical Variance Detection
# =============================================================================


def compute_variance_over_runs(
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    batch: Dict[str, torch.Tensor],
    loss_fn: Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
    n_runs: int = 20,
    seed: int = 42,
    with_sync: bool = True,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Tuple[float, float, List[float]]:
    """
    Run the same step N times and measure variance.

    Reference: Hoffer et al. (2017), Goyal et al. (2017).

    With proper synchronization, variance should be ~0 (deterministic).
    High variance indicates race conditions causing non-determinism.

    Args:
        model_factory: Function to create a fresh model.
        optimizer_factory: Function to create optimizer from model.
        batch: Input batch (will be cloned for each run).
        loss_fn: Loss function.
        n_runs: Number of runs to perform.
        seed: Base random seed.
        with_sync: Whether to synchronize (True = clean, False = may race).
        autocast_dtype: Mixed precision dtype.

    Returns:
        Tuple of (mean_loss, std_loss, all_losses).
    """
    import numpy as np

    losses = []

    for i in range(n_runs):
        set_deterministic_mode(seed)

        model = model_factory()
        optimizer = optimizer_factory(model)

        # Clone batch to ensure identical input
        cloned_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        result = run_reference_step(
            model, optimizer, cloned_batch, loss_fn,
            autocast_dtype=autocast_dtype, with_sync=with_sync
        )
        losses.append(result["loss"])

        # Clean up
        del model, optimizer

    return float(np.mean(losses)), float(np.std(losses)), losses


def verify_variance(
    std_with_sync: float,
    std_without_sync: float,
    variance_ratio_threshold: float = 10.0,
) -> VerificationResult:
    """
    Compare variance with and without sync to detect races.

    Reference: Hoffer et al. (2017) - Deterministic training should have zero variance.

    Args:
        std_with_sync: Standard deviation of losses with synchronization.
        std_without_sync: Standard deviation of losses without synchronization.
        variance_ratio_threshold: Ratio above which indicates race condition.

    Returns:
        VerificationResult indicating if corruption was detected.
    """
    # Add small epsilon to avoid division by zero
    variance_ratio = std_without_sync / (std_with_sync + 1e-12)
    passed = variance_ratio <= variance_ratio_threshold

    if passed:
        message = f"PASSED: Variance ratio={variance_ratio:.2f} (threshold={variance_ratio_threshold})"
    else:
        message = (
            f"FAILED: Variance ratio={variance_ratio:.2f} >> threshold={variance_ratio_threshold} - "
            f"RACE DETECTED! High variance without sync indicates torn reads."
        )

    return VerificationResult(
        method="variance_detection",
        passed=passed,
        details={
            "std_with_sync": std_with_sync,
            "std_without_sync": std_without_sync,
            "variance_ratio": variance_ratio,
            "threshold": variance_ratio_threshold,
        },
        message=message,
    )


# =============================================================================
# Method 3: Gradient Norm Tracking
# =============================================================================


def track_gradient_norms(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    loss_fn: Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
    n_steps: int = 100,
    device: torch.device = None,
    with_sync: bool = True,
    autocast_dtype: Optional[torch.dtype] = None,
) -> List[float]:
    """
    Track gradient norms over multiple steps for outlier detection.

    Reference: Pascanu et al. (2013), Chen et al. (2016).

    Corrupted gradients exhibit abnormally large norms. Track distribution
    and detect outliers.

    Args:
        model: The model to train.
        optimizer: The optimizer.
        dataloader: Training dataloader.
        loss_fn: Loss function.
        n_steps: Number of steps to track.
        device: GPU device.
        with_sync: Whether to synchronize before forward.
        autocast_dtype: Mixed precision dtype.

    Returns:
        List of gradient norms over steps.
    """
    model.train()
    grad_norms = []

    for step, batch in enumerate(dataloader):
        if step >= n_steps:
            break

        # Move to device
        if device is not None:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if with_sync:
            torch.cuda.synchronize()

        optimizer.zero_grad(set_to_none=True)

        if autocast_dtype:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                scores = model(batch)
                loss = loss_fn(scores, batch)
        else:
            scores = model(batch)
            loss = loss_fn(scores, batch)

        loss.backward()

        # Compute total gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        grad_norms.append(total_norm)
        optimizer.step()

    return grad_norms


def verify_gradient_norm_outliers(
    norms_with_sync: List[float],
    norms_without_sync: List[float],
    outlier_sigma: float = 3.0,
    outlier_ratio_threshold: float = 2.0,
) -> VerificationResult:
    """
    Compare gradient norm outlier rates with and without sync.

    Reference: Pascanu et al. (2013), Chen et al. (2016).

    Args:
        norms_with_sync: Gradient norms with synchronization.
        norms_without_sync: Gradient norms without synchronization.
        outlier_sigma: Number of standard deviations for outlier definition.
        outlier_ratio_threshold: Ratio above which indicates race condition.

    Returns:
        VerificationResult indicating if corruption was detected.
    """
    import numpy as np

    # Count outliers in clean run
    mean_sync = np.mean(norms_with_sync)
    std_sync = np.std(norms_with_sync)
    threshold_sync = mean_sync + outlier_sigma * std_sync
    outliers_sync = sum(1 for n in norms_with_sync if n > threshold_sync)
    outlier_rate_sync = outliers_sync / len(norms_with_sync)

    # Count outliers in test run (using clean run's distribution as baseline)
    outliers_no_sync = sum(1 for n in norms_without_sync if n > threshold_sync)
    outlier_rate_no_sync = outliers_no_sync / len(norms_without_sync)

    # Compare rates
    outlier_ratio = outlier_rate_no_sync / (outlier_rate_sync + 1e-6)
    passed = outlier_ratio <= outlier_ratio_threshold

    if passed:
        message = (
            f"PASSED: Outlier ratio={outlier_ratio:.2f} "
            f"(sync={outlier_rate_sync:.1%}, no_sync={outlier_rate_no_sync:.1%})"
        )
    else:
        message = (
            f"FAILED: Outlier ratio={outlier_ratio:.2f} >> threshold={outlier_ratio_threshold} - "
            f"RACE DETECTED! More gradient outliers without sync."
        )

    return VerificationResult(
        method="gradient_norm_outliers",
        passed=passed,
        details={
            "outlier_rate_sync": outlier_rate_sync,
            "outlier_rate_no_sync": outlier_rate_no_sync,
            "outlier_ratio": outlier_ratio,
            "threshold": outlier_ratio_threshold,
            "outliers_sync": outliers_sync,
            "outliers_no_sync": outliers_no_sync,
            "mean_sync": mean_sync,
            "std_sync": std_sync,
        },
        message=message,
    )


# =============================================================================
# Method 4: Instability Detection (Already implemented in inflight_checks.py)
# =============================================================================

# This is the BEST method - see inflight_checks.py
# It detects the race ITSELF (not symptoms) by reading memory multiple times


# =============================================================================
# Combined Verification Runner
# =============================================================================


def run_all_verifications(
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    batch: Dict[str, torch.Tensor],
    loss_fn: Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
    seed: int = 42,
    autocast_dtype: Optional[torch.dtype] = None,
    verbose: bool = True,
) -> Dict[str, VerificationResult]:
    """
    Run all verification methods to check for silent corruption.

    This is the recommended way to verify training correctness when you
    suspect race conditions but don't see NaN.

    Args:
        model_factory: Function to create a fresh model.
        optimizer_factory: Function to create optimizer from model.
        batch: Input batch for testing.
        loss_fn: Loss function.
        seed: Random seed for determinism.
        autocast_dtype: Mixed precision dtype.
        verbose: Whether to log results.

    Returns:
        Dict mapping method names to VerificationResult.
    """
    results = {}

    # Method 1: Deterministic Comparison
    if verbose:
        log.info("Running Method 1: Deterministic Comparison...")

    set_deterministic_mode(seed)
    model_ref = model_factory()
    optimizer_ref = optimizer_factory(model_ref)
    batch_ref = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    reference = run_reference_step(
        model_ref, optimizer_ref, batch_ref, loss_fn,
        autocast_dtype=autocast_dtype, with_sync=True
    )

    set_deterministic_mode(seed)
    model_test = model_factory()
    optimizer_test = optimizer_factory(model_test)
    batch_test = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    test = run_reference_step(
        model_test, optimizer_test, batch_test, loss_fn,
        autocast_dtype=autocast_dtype, with_sync=False  # May have race
    )

    result = verify_deterministic_comparison(reference, test)
    results["deterministic_comparison"] = result
    if verbose:
        log.info("  %s", result.message)

    # Method 2: Variance Detection
    if verbose:
        log.info("Running Method 2: Variance Detection...")

    _, std_sync, _ = compute_variance_over_runs(
        model_factory, optimizer_factory, batch, loss_fn,
        n_runs=10, seed=seed, with_sync=True, autocast_dtype=autocast_dtype
    )
    _, std_no_sync, _ = compute_variance_over_runs(
        model_factory, optimizer_factory, batch, loss_fn,
        n_runs=10, seed=seed, with_sync=False, autocast_dtype=autocast_dtype
    )

    result = verify_variance(std_sync, std_no_sync)
    results["variance_detection"] = result
    if verbose:
        log.info("  %s", result.message)

    # Summary
    if verbose:
        all_passed = all(r.passed for r in results.values())
        if all_passed:
            log.info("ALL VERIFICATIONS PASSED: Training appears correct.")
        else:
            failed = [name for name, r in results.items() if not r.passed]
            log.warning("VERIFICATION FAILED: %s detected corruption!", ", ".join(failed))

    return results


# =============================================================================
# Numerical Stability Verification
# =============================================================================


@dataclass
class StabilityReport:
    """Comprehensive numerical stability report."""

    is_stable: bool
    """Overall stability verdict."""

    is_deterministic: bool
    """Whether training is bitwise reproducible."""

    has_race_condition: bool
    """Whether race condition was detected."""

    has_numerical_issues: bool
    """Whether numerical issues (NaN/Inf, gradient explosion) detected."""

    details: Dict[str, Any] = field(default_factory=dict)
    """Detailed metrics from all checks."""

    recommendations: List[str] = field(default_factory=list)
    """Recommended actions based on findings."""


def check_determinism(
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    batch: Dict[str, torch.Tensor],
    loss_fn: Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
    seed: int = 42,
    autocast_dtype: Optional[torch.dtype] = None,
    n_runs: int = 3,
) -> Tuple[bool, float, List[float]]:
    """
    Check if training is bitwise deterministic.

    Runs the same step multiple times with identical seeds and checks
    if outputs are exactly the same. Non-determinism here indicates
    either race conditions OR non-deterministic CUDA operations.

    Args:
        model_factory: Function to create a fresh model.
        optimizer_factory: Function to create optimizer from model.
        batch: Input batch.
        loss_fn: Loss function.
        seed: Random seed.
        autocast_dtype: Mixed precision dtype.
        n_runs: Number of runs to compare.

    Returns:
        Tuple of (is_deterministic, max_diff, all_losses).
    """
    losses = []

    for _ in range(n_runs):
        set_deterministic_mode(seed)
        torch.cuda.synchronize()  # Ensure clean state

        model = model_factory()
        optimizer = optimizer_factory(model)
        batch_clone = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        result = run_reference_step(
            model, optimizer, batch_clone, loss_fn,
            autocast_dtype=autocast_dtype, with_sync=True
        )
        losses.append(result["loss"])

        del model, optimizer

    # Check if all losses are identical (bitwise)
    max_diff = max(abs(losses[i] - losses[0]) for i in range(1, len(losses)))
    is_deterministic = max_diff == 0.0

    return is_deterministic, max_diff, losses


def check_gradient_health(
    model: nn.Module,
    step: int = 0,
) -> Dict[str, Any]:
    """
    Check gradient health for numerical issues.

    Args:
        model: Model after backward pass.
        step: Current step (for logging).

    Returns:
        Dict with gradient statistics and health flags.
    """
    stats = {
        "step": step,
        "has_nan": False,
        "has_inf": False,
        "has_zero": False,
        "max_norm": 0.0,
        "min_norm": float("inf"),
        "mean_norm": 0.0,
        "total_params": 0,
        "nan_params": [],
        "inf_params": [],
        "zero_params": [],
        "large_grad_params": [],  # Params with unusually large gradients
    }

    norms = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        stats["total_params"] += 1
        grad = param.grad.data

        # Check for NaN
        if torch.isnan(grad).any():
            stats["has_nan"] = True
            stats["nan_params"].append(name)

        # Check for Inf
        if torch.isinf(grad).any():
            stats["has_inf"] = True
            stats["inf_params"].append(name)

        # Compute norm
        norm = grad.norm(2).item()
        norms.append(norm)

        # Check for zero gradients
        if norm == 0.0:
            stats["has_zero"] = True
            stats["zero_params"].append(name)

        # Track large gradients (potential corruption indicator)
        if norm > 1000.0:
            stats["large_grad_params"].append((name, norm))

    if norms:
        stats["max_norm"] = max(norms)
        stats["min_norm"] = min(norms)
        stats["mean_norm"] = sum(norms) / len(norms)

    return stats


def check_loss_health(loss: torch.Tensor) -> Dict[str, Any]:
    """
    Check loss tensor for numerical issues.

    Args:
        loss: Loss tensor.

    Returns:
        Dict with loss health information.
    """
    loss_val = loss.detach()

    return {
        "value": loss_val.item() if loss_val.numel() == 1 else loss_val.mean().item(),
        "has_nan": torch.isnan(loss_val).any().item(),
        "has_inf": torch.isinf(loss_val).any().item(),
        "is_negative": (loss_val < 0).any().item(),
        "is_very_large": (loss_val.abs() > 1e6).any().item(),
    }


def verify_numerical_stability(
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    batch: Dict[str, torch.Tensor],
    loss_fn: Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
    seed: int = 42,
    autocast_dtype: Optional[torch.dtype] = None,
    verbose: bool = True,
) -> StabilityReport:
    """
    Comprehensive numerical stability verification.

    This runs multiple checks to determine:
    1. Is training deterministic? (bitwise reproducible)
    2. Is there a race condition? (different results with/without sync)
    3. Are there numerical issues? (NaN, Inf, gradient explosion)

    Decision tree:
    - Deterministic + same with/without sync → STABLE, no race
    - Deterministic + different with/without sync → RACE CONDITION
    - Non-deterministic + sync doesn't help → Non-deterministic ops (not race)
    - Non-deterministic + sync helps → Likely RACE CONDITION

    Args:
        model_factory: Function to create a fresh model.
        optimizer_factory: Function to create optimizer from model.
        batch: Input batch.
        loss_fn: Loss function.
        seed: Random seed.
        autocast_dtype: Mixed precision dtype.
        verbose: Whether to log progress.

    Returns:
        StabilityReport with diagnosis and recommendations.
    """
    details = {}
    recommendations = []

    if verbose:
        log.info("=" * 60)
        log.info("NUMERICAL STABILITY VERIFICATION")
        log.info("=" * 60)

    # Step 1: Check determinism (with full sync)
    if verbose:
        log.info("Step 1: Checking determinism (with sync)...")

    is_det, max_diff, losses = check_determinism(
        model_factory, optimizer_factory, batch, loss_fn,
        seed=seed, autocast_dtype=autocast_dtype, n_runs=3
    )
    details["determinism_with_sync"] = {
        "is_deterministic": is_det,
        "max_diff": max_diff,
        "losses": losses,
    }

    if verbose:
        if is_det:
            log.info("  ✓ Training IS deterministic with sync (max_diff=0)")
        else:
            log.warning("  ✗ Training is NOT deterministic even with sync (max_diff=%.2e)", max_diff)
            recommendations.append(
                "Non-deterministic operations detected. Check for: "
                "atomicAdd in custom kernels, non-deterministic cuDNN algorithms, "
                "or reduction order differences. Enable torch.use_deterministic_algorithms(True)."
            )

    # Step 2: Check for race condition (compare sync vs no-sync)
    if verbose:
        log.info("Step 2: Checking for race condition (sync vs no-sync)...")

    set_deterministic_mode(seed)
    model_sync = model_factory()
    opt_sync = optimizer_factory(model_sync)
    batch_sync = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    result_sync = run_reference_step(
        model_sync, opt_sync, batch_sync, loss_fn,
        autocast_dtype=autocast_dtype, with_sync=True
    )

    set_deterministic_mode(seed)
    model_nosync = model_factory()
    opt_nosync = optimizer_factory(model_nosync)
    batch_nosync = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    result_nosync = run_reference_step(
        model_nosync, opt_nosync, batch_nosync, loss_fn,
        autocast_dtype=autocast_dtype, with_sync=False  # May have race
    )

    loss_diff = abs(result_sync["loss"] - result_nosync["loss"])
    has_race = loss_diff > 1e-6

    details["race_check"] = {
        "loss_with_sync": result_sync["loss"],
        "loss_without_sync": result_nosync["loss"],
        "loss_diff": loss_diff,
        "has_race": has_race,
    }

    if verbose:
        if has_race:
            log.warning("  ✗ RACE CONDITION DETECTED! Loss differs by %.2e", loss_diff)
            log.warning("    Loss with sync:    %.6f", result_sync["loss"])
            log.warning("    Loss without sync: %.6f", result_nosync["loss"])
            recommendations.append(
                "Race condition detected. Add proper stream synchronization: "
                "torch.cuda.current_stream().wait_stream(memcpy_stream) before forward pass."
            )
        else:
            log.info("  ✓ No race condition detected (loss diff=%.2e)", loss_diff)

    # Step 3: Check gradient health
    if verbose:
        log.info("Step 3: Checking gradient health...")

    # Re-run to get gradient stats
    set_deterministic_mode(seed)
    model_check = model_factory()
    opt_check = optimizer_factory(model_check)
    batch_check = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    model_check.train()
    opt_check.zero_grad()
    torch.cuda.synchronize()

    if autocast_dtype:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            scores = model_check(batch_check)
            loss = loss_fn(scores, batch_check)
    else:
        scores = model_check(batch_check)
        loss = loss_fn(scores, batch_check)

    loss.backward()

    grad_health = check_gradient_health(model_check)
    loss_health = check_loss_health(loss)

    details["gradient_health"] = grad_health
    details["loss_health"] = loss_health

    has_numerical_issues = (
        grad_health["has_nan"] or
        grad_health["has_inf"] or
        loss_health["has_nan"] or
        loss_health["has_inf"]
    )

    if verbose:
        if has_numerical_issues:
            log.warning("  ✗ Numerical issues detected!")
            if grad_health["has_nan"]:
                log.warning("    - NaN in gradients: %s", grad_health["nan_params"][:5])
            if grad_health["has_inf"]:
                log.warning("    - Inf in gradients: %s", grad_health["inf_params"][:5])
            if loss_health["has_nan"]:
                log.warning("    - NaN in loss")
            if loss_health["has_inf"]:
                log.warning("    - Inf in loss")
            recommendations.append(
                "Numerical issues (NaN/Inf) detected. This could be from: "
                "1) Race conditions causing torn reads, "
                "2) Learning rate too high, "
                "3) Gradient explosion, "
                "4) Loss function issues."
            )
        else:
            log.info("  ✓ No NaN/Inf detected in gradients or loss")

        if grad_health["large_grad_params"]:
            log.warning("  ⚠ Large gradients detected (potential instability):")
            for name, norm in grad_health["large_grad_params"][:5]:
                log.warning("    - %s: norm=%.2e", name, norm)

    # Step 4: Summarize
    is_stable = is_det and not has_race and not has_numerical_issues

    if verbose:
        log.info("=" * 60)
        log.info("VERDICT: %s", "STABLE ✓" if is_stable else "UNSTABLE ✗")
        log.info("  - Deterministic: %s", "Yes" if is_det else "No")
        log.info("  - Race condition: %s", "Yes" if has_race else "No")
        log.info("  - Numerical issues: %s", "Yes" if has_numerical_issues else "No")
        if recommendations:
            log.info("RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                log.info("  %d. %s", i, rec)
        log.info("=" * 60)

    return StabilityReport(
        is_stable=is_stable,
        is_deterministic=is_det,
        has_race_condition=has_race,
        has_numerical_issues=has_numerical_issues,
        details=details,
        recommendations=recommendations,
    )


__all__ = [
    "VerificationResult",
    "StabilityReport",
    "set_deterministic_mode",
    "run_reference_step",
    "verify_deterministic_comparison",
    "compute_variance_over_runs",
    "verify_variance",
    "track_gradient_norms",
    "verify_gradient_norm_outliers",
    "run_all_verifications",
    "check_determinism",
    "check_gradient_health",
    "check_loss_health",
    "verify_numerical_stability",
]
