# EXT-4 Full-Pixel Readiness

## Decision

Full-pixel validation is not ready unless `external_full_pixel_ready = 1`.

Current result:

- LaGOT sequences linked: `294`
- Unique LaSOT sequences needed: `280`
- LaGOT re-entry events: `1213`
- Pixel-ready LaSOT sequences: `20`
- Pixel-ready events: `234`
- HuggingFace LaSOT manifest available: `1`

## Download Policy

Do not download full LaSOT automatically. The HuggingFace category zips are multi-GB files.

Use the generated manifest to select categories. Suggested first categories:

`dog, kite, coin, motorcycle, bottle`

Dry-run example:

```powershell
python scripts/download_lasot_hf_categories.py --categories dog
```

Execute example:

```powershell
python scripts/download_lasot_hf_categories.py --categories dog --execute
```

## Next Step

run EXT-4 full-pixel appearance validation on pixel-ready sequences
