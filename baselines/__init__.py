"""Baseline methods for NOPS-OWR."""

from .edge_cluster import EdgeClusterBaseline
from .frame_diff_cc import FrameDiffConnectedComponentsBaseline

__all__ = ["EdgeClusterBaseline", "FrameDiffConnectedComponentsBaseline"]
