# CORE-1AQ Memory Decision API Contract

This contract is intentionally narrow. It does not attach identities, promote tracks, or update prototype heads.

## Inputs

- `query_obs_id`
- online retrieval candidate scores
- online top1 margin
- candidate count / memory candidate metadata

## Outputs

- `retrieval_state = old_recall_candidate`
- `retrieval_state = uncertain_need_more_evidence`

## Rules

- `old_recall_candidate` may be used as a memory retrieval proposal.
- `uncertain_need_more_evidence` must not trigger memory update, attach, promotion, or head update.
- `uncertain_need_more_evidence` may enqueue active evidence acquisition or wait for a more stable future observation.
- GT fields are evaluation-only and must not affect policy action.
