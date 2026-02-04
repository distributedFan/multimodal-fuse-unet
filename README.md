# Multimodal Fuse UNet

本项目包含两个独立的双模态示例：
- RegDB 数据集（热成像 + 可见光）4 类分类，模型为 DualModalUNet。
- Weather 表格数据的双模态 MLP 分类示例，模型为 DualModalMLPNet。

目标是保持逻辑不变的前提下，让结构更清晰、入口更明确。

## 目录结构

```
.
├─ data/
│  ├─ RegDB/                 # RegDB 数据集（热成像/可见光 + idx）
│  └─ dc_weather.csv         # Weather 表格数据
├─ data_prepare.py           # 数据集定义（RegDB + Weather）
├─ configs/                  # 超参数配置（JSON）
├─ model.py                  # 模型定义（DualModalUNet / DualModalMLPNet）
├─ MMUnet.py                 # RegDB 训练入口
├─ MLPnet.py                 # Weather 训练入口
├─ requirements.txt          # 依赖
└─ README.md
```

## 环境依赖

建议 Python 3.9+。

```
pip install -r requirements.txt
```

## 训练入口

RegDB（双模态图像分类）：

```
python MMUnet.py
```

Weather（双模态表格分类）：

```
python MLPnet.py
```

## 配置文件与命令行参数

默认配置放在 `configs/` 下：
- `configs/unet.json`
- `configs/mlp.json`

你可以直接改 JSON，也可以用命令行覆盖：

```
python MMUnet.py --epochs 50 --train-batch-size 32
python MLPnet.py --learning-rate 0.0005 --batch-size 64
```

也可以指定配置文件路径：

```
python MMUnet.py --config configs/unet.json
python MLPnet.py --config configs/mlp.json
```

## 数据说明

- RegDB：使用 `data/RegDB/idx/*.txt` 作为索引文件，代码会自动映射 thermal -> visible 路径，并根据文件名解析四分类标签。
- Weather：使用 `data/dc_weather.csv`，按列索引丢弃无用字段，并将特征一分为二作为两个模态输入。

## 超参数修改位置

- 推荐在 `configs/unet.json` 与 `configs/mlp.json` 中修改。
- 也可以通过命令行参数覆盖对应字段（见上文示例）。

## 备注

- 所有路径均以项目根目录为相对路径。
- 如果你计划扩展为包结构，可将 `model.py` 与 `data_prepare.py` 下沉到 `src/` 并增加 `__init__.py`，训练脚本保持不变即可。
