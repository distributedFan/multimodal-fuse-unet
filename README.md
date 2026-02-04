# Multimodal Fuse UNet

This project provides two independent dual‑modal examples:
- RegDB dataset (thermal + visible) 4‑class classification with DualModalUNet.
- Weather tabular dual‑modal MLP classification with DualModalMLPNet.

The goal is to keep the original logic while making the structure and entry points clearer.

## Project Structure

```
.
├─ data/
│  ├─ RegDB/                 # RegDB dataset (thermal/visible + idx)
│  └─ dc_weather.csv         # Weather tabular data
├─ data_prepare.py           # Dataset definitions (RegDB + Weather)
├─ configs/                  # Hyperparameter configs (JSON)
├─ model.py                  # Model definitions (DualModalUNet / DualModalMLPNet)
├─ MMUnet.py                 # RegDB training entry
├─ MLPnet.py                 # Weather training entry
├─ requirements.txt          # Dependencies
└─ README.md
```

## Requirements

Python 3.9+ is recommended.

```
pip install -r requirements.txt
```

## Training Entry Points

RegDB (dual‑modal image classification):

```
python MMUnet.py
```

Weather (dual‑modal tabular classification):

```
python MLPnet.py
```

## Config Files and CLI Overrides

Default configs are stored in `configs/`:
- `configs/unet.json`
- `configs/mlp.json`

You can edit the JSON directly, or override fields via CLI:

```
python MMUnet.py --epochs 50 --train-batch-size 32
python MLPnet.py --learning-rate 0.0005 --batch-size 64
```

You can also specify a config path explicitly:

```
python MMUnet.py --config configs/unet.json
python MLPnet.py --config configs/mlp.json
```

## Data Notes

- RegDB: uses `data/RegDB/idx/*.txt` index files. The code maps thermal -> visible paths and derives 4‑class labels from filenames.
- Weather: uses `data/dc_weather.csv`, drops unused columns by index, and splits features into two modalities.

## Hyperparameter Changes

- Prefer editing `configs/unet.json` and `configs/mlp.json`.
- CLI args can override any config field (see examples above).

## Notes

- All paths are relative to the project root.
- If you want to refactor into a package layout, move `model.py` and `data_prepare.py` into `src/` and add `__init__.py`. Training scripts can remain unchanged.
