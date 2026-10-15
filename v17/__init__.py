"""V17 — unified 2-player / 4-player Ludo network, trained from scratch.

Plan: docs/V17_CLEAN_PIPELINE_PLAN.md. Rules and the input encoding live in the
Rust crates (ludo_core::unified); this package holds the model, training,
evaluation and tooling. `reference_encoder.py` is an independent pure-Python
implementation of the encoding used only to test the Rust one.
"""
