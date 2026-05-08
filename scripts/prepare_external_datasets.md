# Prepare External Datasets

## Directory Layout

Expected local paths:

```text
data/external/lvos/
data/external/lagot_annotations/
data/external/lasot/
data/external/tao/
data/external/hf_lvosv1_sample/
```

## Public Smoke Sample Already Supported

The EVAL-0 smoke test supports a small HuggingFace LVOS-style point-track sample:

```powershell
New-Item -ItemType Directory -Force -Path data/external/hf_lvosv1_sample
python - <<'PY'
import requests
from pathlib import Path
url = "https://huggingface.co/datasets/allenai/molmo2-single-object-track/resolve/main/lvosv1/train-00000-of-00001.parquet"
out = Path("data/external/hf_lvosv1_sample/train-00000-of-00001.parquet")
out.write_bytes(requests.get(url, timeout=120).content)
PY
```

This file is small and usable for adapter smoke testing. It is not sufficient for final model claims because it contains point trajectories, not full images/masks.

## LaGOT Annotations

LaGOT is a public LaSOT-derived multi-object annotation benchmark. It is useful for oracle-proposal memory-only event mining because it provides real sequence boxes and object identities, but it does not include raw video pixels.

```powershell
git clone --depth 1 https://github.com/google-research-datasets/LaGOT.git data/external/lagot_annotations
```

The raw annotation repository is large because it includes tracker results. Keep it local; do not commit it into this repository. EXT-1 reads:

```text
data/external/lagot_annotations/data/lagot_motchallenge_format.zip
```

## LVOS

Official LVOS distribution may require Google Drive, Kaggle, or Baidu access. Place full LVOS images/masks/annotations under:

```text
data/external/lvos/
```

The current `LVOSAdapter` supports the HuggingFace point-track parquet for smoke testing. Full LVOS image/mask parsing should be added under the same adapter interface.

## LaSOT

Place LaSOT sequences under:

```text
data/external/lasot/<sequence_name>/img/*.jpg
data/external/lasot/<sequence_name>/groundtruth.txt
```

The smoke adapter reads this standard layout.

## TAO

Place TAO annotations and frames under:

```text
data/external/tao/annotations/train.json
data/external/tao/<frame paths referenced by annotations>
```

The smoke adapter reads COCO-style TAO JSON annotations.
