"""Computation of the Halstead complexity measures.

Given the operator/operand multisets produced by
:mod:`ppbcc.code_complexity.classification`, this module derives the classic
Halstead metrics (Maurice Halstead, *Elements of Software Science*, 1977):

===========================  ==================================================
Measure                      Formula
===========================  ==================================================
Vocabulary ``n``             ``n1 + n2``
Length ``N``                 ``N1 + N2``
Calculated length ``N^``     ``n1*log2(n1) + n2*log2(n2)``
Volume ``V``                 ``N * log2(n)``
Difficulty ``D``             ``(n1 / 2) * (N2 / n2)``
Effort ``E``                 ``D * V``
Time ``T``                   ``E / 18`` seconds
Delivered bugs ``B``         ``V / 3000``
Program level ``L``          ``1 / D``
Language level ``lambda``    ``L^2 * V``
===========================  ==================================================

with ``n1``/``n2`` the number of distinct operators/operands and ``N1``/``N2``
their total number of occurrences.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class HalsteadMetrics:
    """The Halstead base counts and derived measures of one program.

    Attributes:
        distinct_operators: Number of distinct operators ``n1``.
        distinct_operands: Number of distinct operands ``n2``.
        total_operators: Total operator occurrences ``N1``.
        total_operands: Total operand occurrences ``N2``.
    """

    distinct_operators: int
    distinct_operands: int
    total_operators: int
    total_operands: int

    @classmethod
    def from_counts(cls, operators: Counter[str], operands: Counter[str]) -> "HalsteadMetrics":
        """Builds the metrics from operator/operand multisets.

        Args:
            operators: Operator occurrences keyed by operator text.
            operands: Operand occurrences keyed by operand text.

        Returns:
            The corresponding :class:`HalsteadMetrics`.
        """
        return cls(
            distinct_operators=len(operators),
            distinct_operands=len(operands),
            total_operators=sum(operators.values()),
            total_operands=sum(operands.values()),
        )

    @property
    def vocabulary(self) -> int:
        """Program vocabulary ``n = n1 + n2``."""
        return self.distinct_operators + self.distinct_operands

    @property
    def length(self) -> int:
        """Program length ``N = N1 + N2``."""
        return self.total_operators + self.total_operands

    @property
    def calculated_length(self) -> float:
        """Estimated program length ``N^ = n1*log2(n1) + n2*log2(n2)``."""
        result = 0.0
        for distinct in (self.distinct_operators, self.distinct_operands):
            if distinct > 0:
                result += distinct * math.log2(distinct)
        return result

    @property
    def volume(self) -> float:
        """Program volume ``V = N * log2(n)`` in bits."""
        if self.vocabulary == 0:
            return 0.0
        return self.length * math.log2(self.vocabulary)

    @property
    def difficulty(self) -> float:
        """Program difficulty ``D = (n1 / 2) * (N2 / n2)``."""
        if self.distinct_operands == 0:
            return 0.0
        return (self.distinct_operators / 2.0) * (self.total_operands / self.distinct_operands)

    @property
    def effort(self) -> float:
        """Programming effort ``E = D * V`` in elementary mental discriminations."""
        return self.difficulty * self.volume

    @property
    def time_seconds(self) -> float:
        """Estimated implementation time ``T = E / 18`` in seconds."""
        return self.effort / 18.0

    @property
    def delivered_bugs(self) -> float:
        """Estimated number of delivered bugs ``B = V / 3000``."""
        return self.volume / 3000.0

    @property
    def program_level(self) -> float:
        """Program level ``L = 1 / D`` (1 is the most abstract program)."""
        if self.difficulty == 0.0:
            return 0.0
        return 1.0 / self.difficulty

    @property
    def language_level(self) -> float:
        """Language level ``lambda = L^2 * V``."""
        return self.program_level**2 * self.volume

    def as_dict(self) -> dict[str, int | float]:
        """Serialises all base counts and derived measures.

        Returns:
            Mapping from metric name (matching the CSV column names) to its
            value, in a stable order.
        """
        return {
            "distinct_operators": self.distinct_operators,
            "distinct_operands": self.distinct_operands,
            "total_operators": self.total_operators,
            "total_operands": self.total_operands,
            "vocabulary": self.vocabulary,
            "length": self.length,
            "calculated_length": self.calculated_length,
            "volume": self.volume,
            "difficulty": self.difficulty,
            "effort": self.effort,
            "time_seconds": self.time_seconds,
            "delivered_bugs": self.delivered_bugs,
            "program_level": self.program_level,
            "language_level": self.language_level,
        }
