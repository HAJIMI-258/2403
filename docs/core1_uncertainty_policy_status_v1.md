# CORE-1 Uncertainty Policy Status v1

## Status

The bounded-wait uncertainty policy is an experimental, disabled-by-default memory decision policy.

It is not an identity attachment policy. It is not a promotion policy. It does not update prototype heads.

## What Changed

The CORE-1 line found that learned descriptor / metric integration is not yet reliable:

- CORE-1AL: learned metric improved top1 slightly, but controls reached the same top1.
- CORE-1AM: the learned metric did not separate from shuffled descriptor perturbations.

The reliable gain came from uncertainty handling:

- CORE-1AP: split-validated margin gate suppressed 6 false old recalls on 495 queries.
- CORE-1AS: bounded wait released 60 / 76 uncertain decisions with 0 wrong releases.
- CORE-1AV: broader 6-sequence regression suppressed 8 false old recalls on 667 queries.
- CORE-1AW: disabled-by-default harness showed precision 0.9505 -> 0.9615 with coverage 0.9730.

## Current Rule

If `policy_enabled = false`:

- Use forced old recall baseline behavior.

If `policy_enabled = true`:

- If top1 margin >= 0.0194, emit `old_recall_candidate`.
- If top1 margin < 0.0194, emit `uncertain_need_more_evidence`.
- Wait up to 10 frames for a high-margin same-track release.
- If no release is found, remain uncertain.

## Safety Constraints

`uncertain_need_more_evidence` must not:

- update memory,
- attach identity,
- promote a track,
- update prototype heads.

## Integration Decision

Allowed:

- Use in evaluation harness as an experimental flag.
- Use as a downstream active-evidence queue source.
- Use as a safety layer to avoid forced false old-object recall.

Not allowed yet:

- Default-enable in main NOPS.
- Treat uncertain decisions as old-object recalls.
- Use it to attach / promote identity.

## Next

Run broader seed/config regression before considering default enablement.
