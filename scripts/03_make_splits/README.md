## 03_make_splits

Run after [`02_embed_and_match`](../02_embed_and_match/), which writes the OT pairs this step reads.

### Train/val split for OT train pairs (`make_splits.py`)

OT matching writes:
- `<idx_root>/<dname>/<dname>_ot_match.train.idx.csv`
- `<idx_root>/<dname>/<dname>_ot_match.test.idx.csv`

This script takes the **train** file and splits it into:
- `<dname>_ot_match.trainsub.idx.csv`
- `<dname>_ot_match.valsub.idx.csv`

Example:

```bash
export DATA_ROOT="/path/to/data_root"
python make_splits.py --dataset k562 --idx_root "$DATA_ROOT/ot_ids/line_specific" --seed 42 --train_frac 0.85
```

Next step: [`04_train`](../04_train/).

