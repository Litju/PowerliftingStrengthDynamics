"""Source adapters.

Each subpackage owns one external source end to end: its provenance, its schema
contract, its acquisition, and its canonical transformation. An adapter must not
leak source semantics into the canonical schema -- the canonical schema states what
PSD can represent, and the adapter is where a source's own meaning is read.
"""
