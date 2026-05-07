# Phase 3D Stage A.3 Recommendation

Do not enter Stage B.

Stage A.3 should be judged on routing only:
- whether cross-lineage preemption is now visible as a tentative state
- whether reroute exposes the proposal to resurrection consume
- whether final assignment can move off the original preempting active lineage

Current minimal policy target result: `was_rerouted=1`, `final_source=resurrection_from_dormant_or_ghost`, `final_lineage=0`.
Forced reroute probe: `restore_attempted_after_reroute=1`, `final_lineage=0`.

Stage A.3 therefore clears the routing-visibility question but does not yet clear lineage-correct recovery after reroute. The next repair should stay inside Stage A and tighten reroute target selection, not move on to promotion or Stage B.
