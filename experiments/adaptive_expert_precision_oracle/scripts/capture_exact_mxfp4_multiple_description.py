#!/usr/bin/env python3
"""Fresh complete-request capture for the multiple-description hypothesis.

The loader and hooks are the already audited exact-MXFP4 capture path.  Only
the prompt corpus and request split differ from the preceding study, ensuring
that the held-out requests were not used to formulate A/B/parity refinement.
"""

from __future__ import annotations

import capture_exact_mxfp4_confirm as capture


capture.PROMPTS = [
    "Explain how copy-on-write memory works and give one situation where it saves substantial memory.",
    "Write a Python function that validates balanced parentheses while ignoring brackets inside quoted strings.",
    "A tank fills at 18 litres per minute and leaks at 3 litres per minute. How long does 450 litres take?",
    "Compare a B-tree and a log-structured merge tree for a write-heavy key-value store.",
    "Why does salt lower the freezing point of water? Explain without using equations first, then give the equation.",
    "Design five adversarial tests for a CSV parser that supports quoted fields and embedded newlines.",
    "Explain calibration and discrimination for a binary classifier using a weather forecast example.",
    "Prove by induction that the sum of the first n odd positive integers is n squared.",
    "Write SQL that reports each customer's longest gap in days between consecutive purchases.",
    "Describe what happens when a filesystem journal is replayed after power loss.",
    "Turn this policy into pseudocode: accept two failures per hour, then open a circuit breaker for ten minutes.",
    "List four threats to causal inference in an observational study and a possible mitigation for each.",
    "Explain why TCP head-of-line blocking differs from HTTP/2 stream-level head-of-line blocking.",
    "Implement an iterative depth-first traversal of a directed graph that detects a cycle.",
    "A biased coin has probability 0.7 of heads. Find the probability of exactly three heads in five tosses.",
    "Contrast eventual consistency and linearizability for a replicated shopping-cart service.",
    "Explain why the night sky is dark despite the enormous number of stars.",
    "Create edge cases for converting signed decimal strings into 32-bit integers without library parsing.",
    "Explain macro-averaged and micro-averaged F1 scores for an imbalanced three-class dataset.",
    "Give a concise contradiction proof that there are infinitely many prime numbers.",
    "Write a relational query using a recursive common-table expression to return every descendant of a node.",
    "Explain how write amplification arises in flash storage and how garbage collection influences latency.",
    "Translate this into pseudocode: process tasks in priority order, preserve FIFO ties, and reject duplicates.",
    "State the assumptions of a difference-in-differences analysis and describe a useful falsification test.",
]


def split_for(index: int) -> str:
    return "train" if index < 12 else ("validation" if index < 18 else "test")


capture.split_for = split_for


if __name__ == "__main__":
    capture.main()
