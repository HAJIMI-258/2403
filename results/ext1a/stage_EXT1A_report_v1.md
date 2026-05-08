# Stage EXT-1A External Failure Analysis

## Verdict

NOPS passive is not failing mainly because the target is absent from memory. It retrieves the target in top5 for most events, but loses top1 under similar distractor competition.

This is still oracle-proposal / geometry-only analysis on LaGOT annotations. Raw LaSOT pixels are not required for this failure diagnosis, but they are required for appearance descriptors and full perception evaluation.

## Compact

```json
{
  "stage": "EXT-1A",
  "event_count": 1213,
  "nops_top1": 0.640560593569662,
  "best_support_trajectory_top1": 0.7485572959604286,
  "nops_vs_best_delta": -0.1079967023907667,
  "nops_top5": 0.9859851607584501,
  "best_only_success_count": 198,
  "nops_only_success_count": 67,
  "nops_top5_but_not_top1_count": 419,
  "nops_main_failure_counts": {
    "success": 777,
    "similar_distractor_confusion": 405,
    "target_not_in_top5": 17,
    "target_in_top5_but_wrong_top1": 14
  },
  "primary_diagnosis": "candidate generation mostly works; top1 selection under similar distractors is the main external failure",
  "raw_pixel_requirement": "not required for this geometry-only failure diagnosis; required before full perception or appearance descriptor claims",
  "next_recommendation": "build geometry-aware NOPS passive calibration or connect LaSOT pixels for appearance/full-pipeline analysis before model changes"
}
```
