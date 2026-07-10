"""Tests for :mod:`code_complexity.halstead`."""

import math
from collections import Counter

import pytest

from code_complexity.halstead import HalsteadMetrics


class TestFromCounts:
    def test_base_counts(self):
        operators = Counter({"+": 2, "=": 1, ";": 3})
        operands = Counter({"a": 2, "b": 1, "1": 1})
        metrics = HalsteadMetrics.from_counts(operators, operands)
        assert metrics.distinct_operators == 3
        assert metrics.distinct_operands == 3
        assert metrics.total_operators == 6
        assert metrics.total_operands == 4


class TestDerivedMeasures:
    """Verifies the formulas on n1=2, n2=2, N1=3, N2=3."""

    @pytest.fixture
    def metrics(self):
        return HalsteadMetrics(
            distinct_operators=2,
            distinct_operands=2,
            total_operators=3,
            total_operands=3,
        )

    def test_vocabulary_and_length(self, metrics):
        assert metrics.vocabulary == 4
        assert metrics.length == 6

    def test_calculated_length(self, metrics):
        assert metrics.calculated_length == pytest.approx(2 * math.log2(2) + 2 * math.log2(2))

    def test_volume(self, metrics):
        assert metrics.volume == pytest.approx(6 * math.log2(4))  # = 12

    def test_difficulty(self, metrics):
        assert metrics.difficulty == pytest.approx((2 / 2) * (3 / 2))  # = 1.5

    def test_effort(self, metrics):
        assert metrics.effort == pytest.approx(1.5 * 12.0)

    def test_time_and_bugs(self, metrics):
        assert metrics.time_seconds == pytest.approx(metrics.effort / 18.0)
        assert metrics.delivered_bugs == pytest.approx(12.0 / 3000.0)

    def test_levels(self, metrics):
        assert metrics.program_level == pytest.approx(1 / 1.5)
        assert metrics.language_level == pytest.approx((1 / 1.5) ** 2 * 12.0)

    def test_as_dict_is_complete_and_ordered(self, metrics):
        data = metrics.as_dict()
        assert list(data)[:4] == [
            "distinct_operators",
            "distinct_operands",
            "total_operators",
            "total_operands",
        ]
        assert data["effort"] == pytest.approx(18.0)


class TestEdgeCases:
    def test_empty_program_yields_zeroes(self):
        metrics = HalsteadMetrics.from_counts(Counter(), Counter())
        for value in metrics.as_dict().values():
            assert value == 0

    def test_no_operands(self):
        metrics = HalsteadMetrics.from_counts(Counter({";": 1, "+": 1}), Counter())
        assert metrics.difficulty == 0.0
        assert metrics.program_level == 0.0
        assert metrics.volume == pytest.approx(2 * math.log2(2))
