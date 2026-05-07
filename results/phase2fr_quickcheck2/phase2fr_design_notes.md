# Phase 2F-R Design Notes

## Part A conclusion carried into Part B

Field-side background response remains the dominant false-positive source, but proposal geometry is still too crude to ignore. The main repair in this pass is not threshold tightening; it is proposal representation repair.

## Proposal path change

- old path: binary objectness mask -> connected component -> coarse enclosing bbox
- new path: binary objectness mask -> connected support region -> support refinement -> refined bbox

## New proposal representation

- each proposal now carries a support region mask
- each proposal keeps both raw component bbox and refined bbox
- ranking is quality-aware instead of score-only, using compactness, fill ratio, aspect penalty, and boundary penalty

## Evaluation slice used here

This comparison run uses shortened Track A / Track C bridge-synthetic sequences as a fast geometry check. The purpose is localization representation repair, not long-horizon tracking evaluation.