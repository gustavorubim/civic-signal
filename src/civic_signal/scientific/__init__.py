"""Scientific verification helpers for M7 CI gates."""

from civic_signal.scientific.contract_parity import ContractParityChecker
from civic_signal.scientific.golden import validate_golden_bundle
from civic_signal.scientific.live_canaries import LiveSourceCanaryRunner
from civic_signal.scientific.mutation import (
    mutation_breaks_check,
    run_standard_mutation_probes,
)
from civic_signal.scientific.parity import numerical_parity_report
from civic_signal.scientific.properties import (
    control_reconciliation_ok,
    covariance_is_psd,
    interval_ordering_ok,
    label_symmetry_holds,
    option_order_invariance_ok,
    probability_simplex_ok,
    run_offline_property_suite,
)

__all__ = [
    "ContractParityChecker",
    "LiveSourceCanaryRunner",
    "control_reconciliation_ok",
    "covariance_is_psd",
    "interval_ordering_ok",
    "label_symmetry_holds",
    "mutation_breaks_check",
    "numerical_parity_report",
    "option_order_invariance_ok",
    "probability_simplex_ok",
    "run_offline_property_suite",
    "run_standard_mutation_probes",
    "validate_golden_bundle",
]
