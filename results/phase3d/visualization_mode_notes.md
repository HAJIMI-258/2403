# Visualization Mode Notes

## Output Mapping

- `matched_lineage_core_case_strip.png`: core recovery chain view.
- `matched_lineage_core_case.gif`: core recovery chain view.
- `proposal_fp_gallery.png`: full debug false-positive view.

## Core Recovery View

- mode: `core_recovery_view`
- only shows boxes directly tied to the current matched-lineage re-entry case
- allowed overlays:
  GT current target
  old track / old prototype object when present
  matched-lineage candidate boxes
  selected attach target
  current head
  temp attach slot proxy box
  promotion candidate

The core recovery view hides unrelated raw proposals. A proposal is shown only if it overlaps the current GT target, belongs to the matched-lineage candidate set, is the selected attach target, is the current head, is the temp attach slot, or is the promotion pending candidate.

## Full Debug View

- mode: `full_debug_view`
- keeps all raw proposals for upstream false-positive inspection
- non-core frames are explicitly labeled `upstream false-proposal frame, not core attach case`

## Interpretation Guardrail

Current GIFs with many boxes do not prove that temporary attach is working. Most of those boxes are upstream raw proposals or false positives. The previous visualization did not filter candidates down to the current core re-entry event, so that GIF cannot be used as evidence that the recovery path is connected.

Do not mix the core recovery view with the false-positive gallery when judging whether the attach path is wired. The first is for recovery-path inspection; the second is only for upstream proposal noise inspection.
