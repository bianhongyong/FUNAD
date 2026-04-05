#!/usr/bin/env python3
"""Tiny demo: sum of squares 1..n."""


def sum_of_squares(n: int) -> int:
    return n * (n + 1) * (2 * n + 1) // 6


if __name__ == "__main__":
    k = 10
    print(f"1²+2²+…+{k}² = {sum_of_squares(k)}")
