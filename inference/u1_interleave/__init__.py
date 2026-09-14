"""End-to-end GeoWeave interleave inference for the geo-aux benchmark.

The model produces text + intermediate auxiliary-line images in a single
``interleave_gen`` call, matching Mode A of the bench. Output is a
predictions.jsonl that downstream L1/L2/L3 evaluators consume directly.
"""
