# Attach State Trace

Per-event comparison between baseline Stage A wiring and forced temp-attach wiring.

## Event 01

- object: `0`
- reappear_frame: `945`
- gap_length: `95`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Forced Temp Attach

- decision made: attach_target=`temporary_attach_slot` action=``
- lineage seed: hint=`3` pre-memory=`3` used=`3`
- state written where: attach_branch=`yes`, attach_written=`yes`, temp_attach=`yes`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`1`

### Diagnosis

- attach state was written, but neither tracker nor continuation path consumed it

## Event 02

- object: `1`
- reappear_frame: `727`
- gap_length: `27`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`1`

### Diagnosis

- matched lineage missing before attach stage

## Event 03

- object: `1`
- reappear_frame: `1418`
- gap_length: `1`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`1`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Diagnosis

- matched lineage missing before attach stage

## Event 04

- object: `2`
- reappear_frame: `40`
- gap_length: `11`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`` pre-memory=`` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`` pre-memory=`` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Diagnosis

- matched lineage missing before attach stage

## Event 05

- object: `2`
- reappear_frame: `383`
- gap_length: `1`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Diagnosis

- matched lineage missing before attach stage

## Event 06

- object: `2`
- reappear_frame: `606`
- gap_length: `6`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`1`

### Diagnosis

- matched lineage missing before attach stage

## Event 07

- object: `2`
- reappear_frame: `990`
- gap_length: `119`

### Baseline

- decision made: attach_target=`current_head` action=``
- lineage seed: hint=`2` pre-memory=`2` used=`2`
- state written where: attach_branch=`yes`, attach_written=`yes`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`1`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`3` pre-memory=`3` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`1`

### Diagnosis

- matched lineage missing before attach stage

## Event 08

- object: `2`
- reappear_frame: `1102`
- gap_length: `1`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`2` pre-memory=`2` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Diagnosis

- matched lineage missing before attach stage

## Event 09

- object: `2`
- reappear_frame: `1106`
- gap_length: `3`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`2` pre-memory=`2` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Diagnosis

- matched lineage missing before attach stage

## Event 10

- object: `2`
- reappear_frame: `1317`
- gap_length: `4`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`3` pre-memory=`3` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`4` pre-memory=`4` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Diagnosis

- matched lineage missing before attach stage

## Event 11

- object: `2`
- reappear_frame: `1319`
- gap_length: `1`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`3` pre-memory=`3` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`4` pre-memory=`4` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Diagnosis

- matched lineage missing before attach stage

## Event 12

- object: `2`
- reappear_frame: `1451`
- gap_length: `6`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`7` pre-memory=`7` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`1`

### Diagnosis

- matched lineage missing before attach stage

## Event 13

- object: `2`
- reappear_frame: `1453`
- gap_length: `1`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`7` pre-memory=`7` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`1`

### Diagnosis

- matched lineage missing before attach stage

## Event 14

- object: `2`
- reappear_frame: `1554`
- gap_length: `2`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`1`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`1` pre-memory=`1` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`1`, same_prototype_final=`1`

### Diagnosis

- matched lineage missing before attach stage

## Event 15

- object: `2`
- reappear_frame: `1715`
- gap_length: `6`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`5` pre-memory=`5` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`1`, continuation_success=`1`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`7` pre-memory=`7` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Diagnosis

- matched lineage missing before attach stage

## Event 16

- object: `2`
- reappear_frame: `1727`
- gap_length: `1`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`3` pre-memory=`3` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`1`, same_prototype_final=`1`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`7` pre-memory=`7` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Diagnosis

- matched lineage missing before attach stage

## Event 17

- object: `2`
- reappear_frame: `1733`
- gap_length: `3`

### Baseline

- decision made: attach_target=`none` action=``
- lineage seed: hint=`3` pre-memory=`3` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`1`, same_prototype_final=`1`

### Forced Temp Attach

- decision made: attach_target=`none` action=``
- lineage seed: hint=`7` pre-memory=`7` used=``
- state written where: attach_branch=`no`, attach_written=`no`, temp_attach=`no`, promotion_pending=`no`
- state consumed where: tracker=`no`, continuation=`no`, restore_attempt=`no`, promotion_step=`no`
- downstream effect: continuation_used=`0`, continuation_success=`0`, same_track_after_attach=`0`, same_track_final=`0`, same_prototype_final=`0`

### Diagnosis

- matched lineage missing before attach stage
