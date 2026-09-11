"""Tests for Harrell's concordance index implementation.

Covers the C-index computation on synthetic survival data with known
concordance properties, edge cases (all censored, single observation,
tied times, tied risks), and the bootstrap confidence interval.

The concordance index measures discrimination: given two subjects, how
often does the model assign higher risk to the one who experienced the
event first?  C = 0.5 is random, C = 1.0 is perfect.
"""

import numpy as np
import pytest

from pte.concordance import (
    ConcordanceResult,
    concordance_bootstrap_ci,
    concordance_index,
)


# ------------------------------------------------------------------
# Perfect and degenerate cases
# ------------------------------------------------------------------
class TestConcordancePerfect:
    def test_perfect_concordance(self) -> None:
        """When risk exactly mirrors event order, C = 1."""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event = np.array([1, 1, 1, 1, 1])
        # Higher risk for earlier events (shorter survival).
        risk = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = concordance_index(time, event, risk)
        assert result.c_index == pytest.approx(1.0)
        assert result.discordant == 0
        assert result.tied_risk == 0

    def test_perfect_discordance(self) -> None:
        """When risk is exactly backwards, C = 0."""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event = np.array([1, 1, 1, 1, 1])
        # Lower risk for earlier events: completely wrong.
        risk = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = concordance_index(time, event, risk)
        assert result.c_index == pytest.approx(0.0)
        assert result.concordant == 0

    def test_constant_risk_gives_half(self) -> None:
        """When all risk scores are equal, C = 0.5 (no discrimination)."""
        time = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.array([1, 1, 1, 1])
        risk = np.array([1.0, 1.0, 1.0, 1.0])
        result = concordance_index(time, event, risk)
        assert result.c_index == pytest.approx(0.5)
        assert result.concordant == 0
        assert result.discordant == 0
        assert result.tied_risk == 6  # C(4,2) = 6 pairs


# ------------------------------------------------------------------
# Censoring
# ------------------------------------------------------------------
class TestConcordanceCensoring:
    def test_all_censored_returns_nan(self) -> None:
        """No usable pairs when all observations are censored."""
        time = np.array([1.0, 2.0, 3.0])
        event = np.array([0, 0, 0])
        risk = np.array([3.0, 2.0, 1.0])
        result = concordance_index(time, event, risk)
        assert np.isnan(result.c_index)
        assert result.usable == 0

    def test_censored_shorter_time_excluded(self) -> None:
        """A pair where the shorter time is censored should be excluded."""
        # Subject 0: time=1, censored.  Subject 1: time=3, event.
        # The pair (0,1) is NOT usable because subject 0 is censored.
        time = np.array([1.0, 3.0])
        event = np.array([0, 1])
        risk = np.array([5.0, 1.0])
        result = concordance_index(time, event, risk)
        assert result.usable == 0
        assert np.isnan(result.c_index)

    def test_censored_longer_time_still_usable(self) -> None:
        """A pair where the shorter time has an event is usable even
        if the longer time is censored."""
        # Subject 0: time=1, event.  Subject 1: time=3, censored.
        # Usable because subject 0 (shorter) had the event.
        time = np.array([1.0, 3.0])
        event = np.array([1, 0])
        risk = np.array([5.0, 1.0])
        result = concordance_index(time, event, risk)
        assert result.usable == 1
        assert result.concordant == 1
        assert result.c_index == pytest.approx(1.0)

    def test_partial_censoring_pair_counts(self) -> None:
        """Verify pair counts with a mix of events and censoring."""
        time = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.array([1, 0, 1, 1])
        risk = np.array([4.0, 3.0, 2.0, 1.0])
        result = concordance_index(time, event, risk)
        # Usable pairs: (0,1) no (1 is censored at longer time, but
        #   shorter-time subject 0 had event -> usable!  Wait:
        #   time[0]=1 < time[1]=2, event[0]=1 -> usable, concordant
        # (0,2): time[0]=1 < time[2]=3, event[0]=1 -> usable, concordant
        # (0,3): time[0]=1 < time[3]=4, event[0]=1 -> usable, concordant
        # (1,2): time[1]=2 < time[2]=3, event[1]=0 -> NOT usable
        # (1,3): time[1]=2 < time[3]=4, event[1]=0 -> NOT usable
        # (2,3): time[2]=3 < time[3]=4, event[2]=1 -> usable, concordant
        assert result.usable == 4
        assert result.concordant == 4
        assert result.c_index == pytest.approx(1.0)


# ------------------------------------------------------------------
# Tied times
# ------------------------------------------------------------------
class TestConcordanceTiedTimes:
    def test_tied_times_both_events_tied_risk(self) -> None:
        """Tied event times with equal risk count as tied_risk."""
        time = np.array([2.0, 2.0])
        event = np.array([1, 1])
        risk = np.array([1.0, 1.0])
        result = concordance_index(time, event, risk)
        assert result.tied_risk == 1
        assert result.usable == 1
        assert result.c_index == pytest.approx(0.5)

    def test_tied_times_one_censored_excluded(self) -> None:
        """Tied times where one is censored: pair is skipped."""
        time = np.array([2.0, 2.0])
        event = np.array([1, 0])
        risk = np.array([5.0, 1.0])
        result = concordance_index(time, event, risk)
        assert result.usable == 0


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------
class TestConcordanceEdgeCases:
    def test_single_observation_returns_nan(self) -> None:
        result = concordance_index(
            np.array([1.0]), np.array([1]), np.array([0.5])
        )
        assert np.isnan(result.c_index)

    def test_empty_arrays_returns_nan(self) -> None:
        result = concordance_index(
            np.array([]), np.array([]), np.array([])
        )
        assert np.isnan(result.c_index)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            concordance_index(
                np.array([1.0, 2.0]),
                np.array([1]),
                np.array([0.5, 0.3]),
            )

    def test_nan_values_dropped(self) -> None:
        """NaN entries should be silently dropped."""
        time = np.array([1.0, np.nan, 3.0, 4.0])
        event = np.array([1, 1, 1, 1])
        risk = np.array([4.0, 3.0, 2.0, 1.0])
        result = concordance_index(time, event, risk)
        # After dropping index 1, we have 3 subjects with perfect ranking.
        assert result.c_index == pytest.approx(1.0)
        assert result.usable == 3  # C(3,2) = 3


# ------------------------------------------------------------------
# Somers' D
# ------------------------------------------------------------------
class TestSomersD:
    def test_perfect_concordance_somers_d(self) -> None:
        """C = 1 implies D = 1."""
        result = ConcordanceResult(
            c_index=1.0, concordant=10, discordant=0, tied_risk=0, usable=10
        )
        assert result.somers_d == pytest.approx(1.0)

    def test_random_concordance_somers_d(self) -> None:
        """C = 0.5 implies D = 0."""
        result = ConcordanceResult(
            c_index=0.5, concordant=5, discordant=5, tied_risk=0, usable=10
        )
        assert result.somers_d == pytest.approx(0.0)

    def test_perfect_discordance_somers_d(self) -> None:
        """C = 0 implies D = -1."""
        result = ConcordanceResult(
            c_index=0.0, concordant=0, discordant=10, tied_risk=0, usable=10
        )
        assert result.somers_d == pytest.approx(-1.0)


# ------------------------------------------------------------------
# Bootstrap CI
# ------------------------------------------------------------------
class TestConcordanceBootstrapCI:
    def test_ci_contains_point_estimate(self) -> None:
        """The confidence interval should contain the point estimate."""
        rng = np.random.default_rng(42)
        n = 100
        time = rng.exponential(5.0, size=n)
        event = rng.binomial(1, 0.7, size=n)
        risk = -time + rng.normal(0, 1, size=n)  # correlated with time

        ci = concordance_bootstrap_ci(time, event, risk, n_bootstrap=200, seed=0)
        assert ci["ci_low"] <= ci["c_index"] <= ci["ci_high"]

    def test_ci_width_decreases_with_alpha(self) -> None:
        """A wider alpha should produce a narrower interval."""
        rng = np.random.default_rng(7)
        n = 80
        time = rng.exponential(3.0, size=n)
        event = np.ones(n)
        risk = -time + rng.normal(0, 0.5, size=n)

        ci_95 = concordance_bootstrap_ci(
            time, event, risk, n_bootstrap=300, alpha=0.05, seed=1
        )
        ci_80 = concordance_bootstrap_ci(
            time, event, risk, n_bootstrap=300, alpha=0.20, seed=1
        )
        width_95 = ci_95["ci_high"] - ci_95["ci_low"]
        width_80 = ci_80["ci_high"] - ci_80["ci_low"]
        assert width_80 < width_95

    def test_se_is_positive(self) -> None:
        rng = np.random.default_rng(99)
        n = 50
        time = rng.exponential(2.0, size=n)
        event = np.ones(n)
        risk = rng.normal(size=n)

        ci = concordance_bootstrap_ci(time, event, risk, n_bootstrap=100, seed=2)
        assert ci["se"] > 0.0

    def test_n_bootstrap_ok_reported(self) -> None:
        """All bootstrap samples should succeed for uncensored data."""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event = np.ones(5)
        risk = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        ci = concordance_bootstrap_ci(time, event, risk, n_bootstrap=50, seed=0)
        assert ci["n_bootstrap_ok"] > 0
        assert ci["n_bootstrap_ok"] <= 50
