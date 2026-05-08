from __future__ import annotations

from election_outcomes.models.polling_kalman import KalmanPollingModel


class PollingModel(KalmanPollingModel):
    """Stable polling component facade backed by the deterministic Kalman model."""
