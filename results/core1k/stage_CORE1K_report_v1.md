# CORE-1K Windowed Render Cache Smoke

This stage renders only the selected CORE-1J event windows and extracts same-space crop descriptors. It does not execute the full tracker and does not write raw frame caches.

## Result

- Selected sequences: 3
- Selected windows: 12
- Processed events with disappear/reappear descriptors: 6
- Descriptor available rate: 1.0000
- Mean same-event margin: 0.039520
- Same-event top1 rate: 0.8333
- Random top1 rate: 0.3611
- Runtime seconds: 180.29
- Fallback pre-disappear windows used: 3
- Smoke passed: 1

## Interpretation

CORE-1K is a runtime and descriptor separability gate. A pass means the selected rendered windows contain enough same-space descriptor signal to justify a windowed tracker-pair mining pass. A fail means the window descriptor path should be inspected before spending time on tracker execution.

Next recommendation: CORE-1L windowed tracker pair mining on selected rendered windows
