from __future__ import annotations

import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import FrameType
from typing import Any, ClassVar, Generic, TypeVar

T = TypeVar("T")


class BayesianTimeoutError(TimeoutError):
    """Raised when Bayesian inference exceeds the configured wall clock budget."""


class PreviousPosteriorCompatibilityError(ValueError):
    """Raised when a readable previous-posterior artifact is not reusable."""


class FailoverRefusedError(RuntimeError):
    """Raised when the ordered failover policy reaches its literal refuse path."""

    def __init__(self, message: str, audit: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit = audit


@dataclass(frozen=True)
class FailoverPolicy:
    PREVIOUS_POSTERIOR: ClassVar[str] = "previous_posterior_reuse"
    ANALYTIC: ClassVar[str] = "analytic_logit_normal_fallback"
    KALMAN: ClassVar[str] = "kalman_fallback"
    REFUSE: ClassVar[str] = "refuse"
    SUPPORTED_FALLBACKS: ClassVar[tuple[str, ...]] = (
        PREVIOUS_POSTERIOR,
        ANALYTIC,
        KALMAN,
        REFUSE,
    )
    timeout_seconds: float | None = None
    fallback_order: tuple[str, ...] = ("analytic_logit_normal_fallback",)
    block_publication_on_fallback: bool = True

    def __post_init__(self) -> None:
        if len(set(self.fallback_order)) != len(self.fallback_order):
            raise ValueError("Bayesian fallback_order must not contain duplicate paths")
        unsupported = [
            fallback for fallback in self.fallback_order if fallback not in self.SUPPORTED_FALLBACKS
        ]
        if unsupported:
            raise ValueError("Unimplemented Bayesian fallback path(s): " + ", ".join(unsupported))
        if self.REFUSE in self.fallback_order and self.fallback_order[-1] != self.REFUSE:
            raise ValueError("The literal refuse fallback must be the final ordered path")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FailoverPolicy:
        bayesian = dict(config.get("bayesian", {}))
        nuts = dict(bayesian.get("nuts", {}))
        failover = dict(nuts.get("failover", config.get("failover", {})))
        raw_order = failover.get("fallback_order", cls.fallback_order)
        if isinstance(raw_order, str):
            order = tuple(part.strip() for part in raw_order.split(",") if part.strip())
        elif isinstance(raw_order, (list, tuple)):
            order = tuple(str(item) for item in raw_order if str(item).strip())
        else:
            order = cls.fallback_order
        timeout = nuts.get("wall_clock_timeout_seconds", failover.get("timeout_seconds"))
        return cls(
            timeout_seconds=float(timeout) if timeout is not None else None,
            fallback_order=order or cls.fallback_order,
            block_publication_on_fallback=bool(failover.get("block_publication_on_fallback", True)),
        )

    def with_timeout(self, timeout_seconds: float) -> FailoverPolicy:
        return replace(self, timeout_seconds=float(timeout_seconds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "fallback_order": list(self.fallback_order),
            "block_publication_on_fallback": self.block_publication_on_fallback,
        }


@dataclass(frozen=True)
class FailoverResult:
    result: Any
    audit: dict[str, Any]


@dataclass(frozen=True)
class LoadedPreviousPosterior(Generic[T]):
    """A disk-loaded previous posterior plus its explicit compatibility decision."""

    result: T | None
    artifact_path: str
    loaded: bool
    compatible: bool
    reason: str
    metadata: dict[str, Any]


def load_previous_posterior_artifact(
    path: str | Path,
    *,
    loader: Callable[[Path], T],
    validator: Callable[[T], tuple[bool, str, dict[str, Any]]],
) -> LoadedPreviousPosterior[T]:
    """Load and validate a previous posterior without silently accepting corruption."""

    artifact_path = Path(path)
    try:
        result = loader(artifact_path)
    except PreviousPosteriorCompatibilityError as exc:
        return LoadedPreviousPosterior(
            result=None,
            artifact_path=str(artifact_path),
            loaded=True,
            compatible=False,
            reason=str(exc),
            metadata={},
        )
    except Exception as exc:
        return LoadedPreviousPosterior(
            result=None,
            artifact_path=str(artifact_path),
            loaded=False,
            compatible=False,
            reason=f"artifact load failed: {exc}",
            metadata={},
        )
    try:
        compatible, reason, metadata = validator(result)
    except Exception as exc:
        compatible, reason, metadata = False, f"artifact validation failed: {exc}", {}
    return LoadedPreviousPosterior(
        result=result if compatible else None,
        artifact_path=str(artifact_path),
        loaded=True,
        compatible=bool(compatible),
        reason=str(reason),
        metadata=dict(metadata),
    )


def dispatch_failover(
    policy: FailoverPolicy,
    *,
    primary_engine: str,
    reason: str,
    handlers: dict[str, Callable[[], T]] | None = None,
    previous_posterior: LoadedPreviousPosterior[T] | None = None,
    elapsed_seconds: float | None = None,
) -> FailoverResult:
    """Execute the first literally available path in the configured failover order."""

    handlers = dict(handlers or {})
    attempts: list[dict[str, Any]] = []
    for label in policy.fallback_order:
        if label == policy.PREVIOUS_POSTERIOR:
            if previous_posterior is None:
                attempts.append(
                    {"path": label, "status": "unavailable", "reason": "no artifact loaded"}
                )
                continue
            if not previous_posterior.loaded or not previous_posterior.compatible:
                attempts.append(
                    {
                        "path": label,
                        "status": "incompatible"
                        if previous_posterior.loaded
                        else "corrupt_or_unreadable",
                        "reason": previous_posterior.reason,
                        "artifact_path": previous_posterior.artifact_path,
                    }
                )
                continue
            if previous_posterior.result is None:
                attempts.append(
                    {
                        "path": label,
                        "status": "incompatible",
                        "reason": "compatible artifact did not contain a reusable result",
                        "artifact_path": previous_posterior.artifact_path,
                    }
                )
                continue
            result = previous_posterior.result
            attempts.append(
                {
                    "path": label,
                    "status": "executed",
                    "artifact_path": previous_posterior.artifact_path,
                    "metadata": previous_posterior.metadata,
                }
            )
            return _fallback_result(
                result,
                policy=policy,
                primary_engine=primary_engine,
                label=label,
                reason=reason,
                attempts=attempts,
                elapsed_seconds=elapsed_seconds,
            )
        if label == policy.REFUSE:
            attempts.append({"path": label, "status": "executed", "reason": reason})
            audit = _failover_audit(
                policy=policy,
                primary_engine=primary_engine,
                status="refused",
                fallback_used=None,
                executed_path=label,
                reason=reason,
                attempts=attempts,
                elapsed_seconds=elapsed_seconds,
                publication_blocked=True,
            )
            raise FailoverRefusedError(
                f"Bayesian inference failed and failover policy refused publication: {reason}",
                audit,
            )
        handler = handlers.get(label)
        if handler is None:
            attempts.append(
                {"path": label, "status": "unavailable", "reason": "handler not configured"}
            )
            continue
        try:
            result = handler()
        except Exception as exc:
            attempts.append({"path": label, "status": "failed", "reason": str(exc)})
            continue
        attempts.append({"path": label, "status": "executed"})
        return _fallback_result(
            result,
            policy=policy,
            primary_engine=primary_engine,
            label=label,
            reason=reason,
            attempts=attempts,
            elapsed_seconds=elapsed_seconds,
        )

    audit = _failover_audit(
        policy=policy,
        primary_engine=primary_engine,
        status="refused",
        fallback_used=None,
        executed_path=policy.REFUSE,
        reason=f"{reason}; no configured fallback path executed",
        attempts=attempts,
        elapsed_seconds=elapsed_seconds,
        publication_blocked=True,
    )
    raise FailoverRefusedError(
        "Bayesian inference failed and no configured fallback path was executable",
        audit,
    )


def _fallback_result(
    result: T,
    *,
    policy: FailoverPolicy,
    primary_engine: str,
    label: str,
    reason: str,
    attempts: list[dict[str, Any]],
    elapsed_seconds: float | None,
) -> FailoverResult:
    return FailoverResult(
        result=result,
        audit=_failover_audit(
            policy=policy,
            primary_engine=primary_engine,
            status="fallback_used",
            fallback_used=label,
            executed_path=label,
            reason=reason,
            attempts=attempts,
            elapsed_seconds=elapsed_seconds,
            publication_blocked=policy.block_publication_on_fallback,
        ),
    )


def _failover_audit(
    *,
    policy: FailoverPolicy,
    primary_engine: str,
    status: str,
    fallback_used: str | None,
    executed_path: str,
    reason: str,
    attempts: list[dict[str, Any]],
    elapsed_seconds: float | None,
    publication_blocked: bool,
) -> dict[str, Any]:
    return {
        "status": status,
        "primary_engine": primary_engine,
        "fallback_used": fallback_used,
        "executed_path": executed_path,
        "reason": reason,
        "elapsed_seconds": round(float(elapsed_seconds), 6)
        if elapsed_seconds is not None
        else None,
        "timeout_seconds": policy.timeout_seconds,
        "fallback_order": list(policy.fallback_order),
        "attempts": attempts,
        "publication_blocked": bool(publication_blocked),
        "quarantine_required": bool(publication_blocked),
    }


def execute_with_failover(
    primary: Callable[[], T],
    fallback: Callable[[], T] | None,
    policy: FailoverPolicy,
    *,
    primary_engine: str,
    handlers: dict[str, Callable[[], T]] | None = None,
    previous_posterior: LoadedPreviousPosterior[T] | None = None,
    dispatch_on_timeout: bool | None = None,
) -> FailoverResult:
    """Run a primary inference callable under a wall-clock timeout.

    The function intentionally only catches timeout failures. Model exceptions still
    surface as implementation bugs unless the caller explicitly converts them into a
    fallback decision. On timeout the ordered dispatcher is invoked when a fallback
    surface is configured (callable fallback, handlers, previous posterior, or an
    explicit ``dispatch_on_timeout=True``) so negative-matrix audits retain attempt
    status for timeout → unavailable / incompatible / executed / refused.
    """

    started = time.perf_counter()
    try:
        with _wall_clock_timeout(policy.timeout_seconds):
            result = primary()
    except BayesianTimeoutError as exc:
        elapsed = time.perf_counter() - started
        should_dispatch = (
            dispatch_on_timeout
            if dispatch_on_timeout is not None
            else (fallback is not None or handlers is not None or previous_posterior is not None)
        )
        if not should_dispatch:
            raise
        first = policy.fallback_order[0] if policy.fallback_order else policy.REFUSE
        dispatch_handlers = dict(handlers or {})
        if (
            fallback is not None
            and first not in {policy.PREVIOUS_POSTERIOR, policy.REFUSE}
            and first not in dispatch_handlers
        ):
            dispatch_handlers[first] = fallback
        return dispatch_failover(
            policy,
            primary_engine=primary_engine,
            reason=str(exc),
            handlers=dispatch_handlers,
            previous_posterior=previous_posterior,
            elapsed_seconds=elapsed,
        )
    elapsed = time.perf_counter() - started
    return FailoverResult(
        result=result,
        audit={
            "status": "completed",
            "primary_engine": primary_engine,
            "fallback_used": None,
            "executed_path": primary_engine,
            "reason": None,
            "elapsed_seconds": round(float(elapsed), 6),
            "timeout_seconds": policy.timeout_seconds,
            "fallback_order": list(policy.fallback_order),
            "attempts": [],
            "publication_blocked": False,
            "quarantine_required": False,
        },
    )


def exercise_timeout_failover(
    policy: FailoverPolicy,
    *,
    previous_posterior: LoadedPreviousPosterior[Any] | None = None,
    handlers: dict[str, Callable[[], Any]] | None = None,
) -> dict[str, Any]:
    """Run a deterministic fixture audit that forces the first fallback path."""

    audit_policy = policy.with_timeout(0.01)

    def primary() -> str:
        time.sleep(0.05)
        return "primary_completed"

    expected = policy.fallback_order[0] if policy.fallback_order else policy.REFUSE

    def fallback() -> str:
        return str(expected)

    try:
        result = execute_with_failover(
            primary,
            fallback if expected not in {policy.PREVIOUS_POSTERIOR, policy.REFUSE} else None,
            audit_policy,
            primary_engine="numpyro-nuts-fixture",
            handlers=handlers,
            previous_posterior=previous_posterior,
            # Always dispatch so refuse / unavailable previous-posterior paths leave audits.
            dispatch_on_timeout=True,
        )
    except FailoverRefusedError as exc:
        audit = dict(exc.audit)
        return {
            "status": "exercised",
            "passed": audit.get("status") == "refused"
            and audit.get("executed_path") == policy.REFUSE
            and audit.get("publication_blocked") is True
            and isinstance(audit.get("attempts"), list)
            and bool(audit.get("attempts")),
            "audit_scope": "forced_timeout_fixture_not_forecast_fallback",
            "result": None,
            "policy": policy.to_dict(),
            "audit": audit,
        }
    except BayesianTimeoutError as exc:
        return {
            "status": "exercised",
            "passed": False,
            "audit_scope": "forced_timeout_fixture_not_forecast_fallback",
            "result": None,
            "policy": policy.to_dict(),
            "audit": {
                "status": "timeout_unhandled",
                "primary_engine": "numpyro-nuts-fixture",
                "fallback_used": None,
                "executed_path": None,
                "reason": str(exc),
                "fallback_order": list(policy.fallback_order),
                "attempts": [],
                "publication_blocked": True,
                "quarantine_required": True,
            },
        }

    audit = result.audit
    publication_ok = (
        audit.get("publication_blocked") is True
        if policy.block_publication_on_fallback
        else audit.get("publication_blocked") is False
    )
    return {
        "status": "exercised",
        "passed": (
            audit.get("fallback_used") == expected
            and result.result is not None
            and publication_ok
            and isinstance(audit.get("attempts"), list)
            and bool(audit.get("attempts"))
        ),
        "audit_scope": "forced_timeout_fixture_not_forecast_fallback",
        "result": result.result,
        "policy": policy.to_dict(),
        "audit": audit,
    }


@contextmanager
def _wall_clock_timeout(timeout_seconds: float | None) -> Iterator[None]:
    if timeout_seconds is None or timeout_seconds <= 0:
        yield
        return
    if not hasattr(signal, "setitimer"):
        yield
        return

    def _raise_timeout(_signum: int, _frame: FrameType | None) -> None:
        raise BayesianTimeoutError(
            f"Bayesian inference exceeded {float(timeout_seconds):.3f}s wall-clock timeout"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
