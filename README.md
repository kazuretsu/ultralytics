# Ultralytics YOLO26 with Lightweight Feature Extractors

A fork of [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) that keeps the YOLO26 neck and detection head but lets you replace its backbone with a pretrained, mobile-first image classifier from [timm](https://github.com/huggingface/pytorch-image-models): **MobileNetV4** or **EfficientNetV2**.

The target is detection that runs **offline on low-resource hardware**: Android phones from 2020 onward (exported to LiteRT, formerly TensorFlow Lite) and laptops with a 6 GB GPU. Everything else in Ultralytics — training loop, augmentation, validation, export — works unchanged.

- **Upstream sync:** Ultralytics **v8.4.163**
- **License:** AGPL-3.0, same as upstream (see [License](#license))
- **Full Ultralytics documentation:** <https://docs.ultralytics.com>

---

## Contents

1. [The three model options](#1-the-three-model-options)
2. [What this fork changes](#2-what-this-fork-changes)
3. [Install](#3-install)
4. [Quick start](#4-quick-start)
5. [Training on Kaggle](#5-training-on-kaggle)
6. [Nano or small?](#6-nano-or-small)
7. [How the backbone swap works](#7-how-the-backbone-swap-works)
8. [Adding another feature extractor](#8-adding-another-feature-extractor)
9. [Offline and low-resource deployment](#9-offline-and-low-resource-deployment)
10. [Troubleshooting](#10-troubleshooting)
11. [Other fork changes](#11-other-fork-changes)
12. [License](#license)

---

## 1. The three model options

Every option uses the same YOLO26 nano neck and NMS-free detection head. Only the feature extractor changes.

| Load this                           | Backbone                                          | Params | GFLOPs @640 | Starts from                             |
| ----------------------------------- | ------------------------------------------------- | ------ | ----------- | --------------------------------------- |
| `yolo26n.pt`                        | Stock YOLO26                                      | 2.57M  | 6.2         | Whole detector pretrained on COCO       |
| `yolo26n-mobilenetv4convsmall.yaml` | MobileNetV4-Conv-Small (`mobilenetv4_conv_small`) | 3.66M  | 7.1         | ImageNet backbone, random neck and head |
| `yolo26n-efficientnetv2b0.yaml`     | EfficientNetV2-B0 (`tf_efficientnetv2_b0`)        | 7.18M  | 15.1        | ImageNet backbone, random neck and head |

The two backbone configs live in `ultralytics/cfg/models/26/` as `yolo26-mobilenetv4convsmall.yaml` and `yolo26-efficientnetv2b0.yaml`. The file name carries the **backbone variant**; the **head size** comes from the scale letter you add when loading, so `yolo26n-efficientnetv2b0.yaml` means "YOLO26 nano head on EfficientNetV2-B0". That exact name is what shows up in commands, run folders and model summaries, and the same file also builds `yolo26s-efficientnetv2b0.yaml` without a second copy.

**Larger EfficientNetV2:** `yolo26-efficientnetv2s.yaml` uses EfficientNetV2-S (`tf_efficientnetv2_s`, ImageNet-21k weights fine-tuned on ImageNet-1k) for when B0's features are the limit. Load it as `yolo26n-efficientnetv2s.yaml`. It is 21.47M parameters and 50.4 GFLOPs, about 3× the B0 model (costs in [section 6](#6-nano-or-small)). For a size-matched MobileNetV4 comparison, pair it with `yolo26-mobilenetv4convmedium.yaml`, not Conv-Small.

**Larger MobileNetV4:** `yolo26-mobilenetv4convmedium.yaml` uses MobileNetV4-Conv-Medium (`mobilenetv4_conv_medium`) for when Conv-Small's features are the limit. Load it as `yolo26n-mobilenetv4convmedium.yaml`. It is 9.61M parameters and 17.9 GFLOPs, between EfficientNetV2-B0 (7.18M, 15.1) and EfficientNetV2-S (21.47M, 50.4). Like Conv-Small it is all convolutions, with no attention blocks in the backbone. Its LiteRT latency is not in [section 6](#6-nano-or-small) yet.

**A note for comparisons:** `yolo26n.pt` starts with its neck and head already trained on COCO; the backbone configs do not. For a comparison where architecture is the only difference, also train `yolo26n.yaml` — the same stock model with no COCO weights.

### Check that a checkpoint really is YOLO26

A model built from a YAML is YOLO26 only if the YAML says `end2end: True` and `reg_max: 1`. Without those lines `parse_model` falls back to the YOLOv8/YOLO11-style head (16 DFL bins, NMS required) no matter what the file is called. To check any `best.pt`:

```python
from ultralytics import YOLO

m = YOLO("best.pt")
head = m.model.model[-1]
print("reg_max:", head.reg_max, "| NMS-free head:", hasattr(head, "one2one_cv2"))  # YOLO26: 1 | True
print("started from:", m.ckpt.get("train_args", {}).get("model"))
```

---

## 2. What this fork changes

| File                                                          | Change                                                                                                  |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `ultralytics/nn/modules/block.py`                             | `MultiScaleBackbone` base class, `TimmBackbone` and `FeatureSelect`                                     |
| `ultralytics/nn/modules/__init__.py`                          | Re-exports the new modules from the package                                                             |
| `ultralytics/nn/tasks.py`                                     | Imports the backbones and teaches `parse_model()` to handle a layer that returns _several_ feature maps |
| `ultralytics/cfg/models/26/yolo26-mobilenetv4convsmall.yaml`  | YOLO26 + MobileNetV4-Conv-Small config                                                                  |
| `ultralytics/cfg/models/26/yolo26-efficientnetv2b0.yaml`      | YOLO26 + EfficientNetV2-B0 config                                                                       |
| `ultralytics/cfg/models/26/yolo26-efficientnetv2s.yaml`       | YOLO26 + EfficientNetV2-S config (larger backbone)                                                      |
| `ultralytics/cfg/models/26/yolo26-mobilenetv4convmedium.yaml` | YOLO26 + MobileNetV4-Conv-Medium config (larger backbone)                                               |
| `ultralytics/utils/plotting.py`                               | Normalizes box corner order in `Annotator.box_label` (see [Other fork changes](#11-other-fork-changes)) |
| `pyproject.toml`                                              | `backbones` optional extra that pulls in `timm`                                                         |

Nothing else in the library was modified, so upstream releases can still be merged in.

---

## 3. Install

```bash
git clone https://github.com/kazuretsu/ultralytics.git
cd ultralytics
pip install -e ".[backbones]"                # editable install + timm
pip install -e ".[backbones,export-litert]"  # ...plus the LiteRT exporter for Android (Linux x86-64 or macOS, Python >= 3.10)
```

`timm` is imported lazily inside `TimmBackbone`, so `import ultralytics` and the stock YOLO26 models work without it. You need it for the two backbone configs.

---

## 4. Quick start

```python
from ultralytics import YOLO

model = YOLO("yolo26n-mobilenetv4convsmall.yaml")  # or yolo26n-efficientnetv2b0.yaml, or yolo26n.pt
model.train(data="my_data.yaml", epochs=150, imgsz=640, batch=16)
model.val()
model.export(format="litert", nms=False)  # Android; see section 9 for quantization and laptop GPUs
```

Or from the CLI:

```bash
yolo detect train model=yolo26n-efficientnetv2b0.yaml data=my_data.yaml epochs=150 imgsz=640
```

**Start the backbone configs from the `.yaml`, never a `.pt`.** No pretrained YOLO checkpoint exists for a swapped backbone: the feature extractor loads its ImageNet weights and the neck and head start random, so plan for a longer schedule than fine-tuning `yolo26n.pt`.

---

## 5. Training on Kaggle

In the notebook's settings, pick a **GPU** accelerator and turn **Internet on** (needed for `pip`, `yolo26n.pt`, and the ImageNet backbone weights from the Hugging Face Hub). Then:

```python
!pip install -q "ultralytics[backbones] @ git+https://github.com/kazuretsu/ultralytics.git@feat/optimized-inference"
!ls /kaggle/input   # attached Kaggle datasets are mounted here, read-only
```

Point a data YAML at the attached dataset. If the dataset already ships one, copy it and fix `path:`; otherwise write it:

```python
from pathlib import Path

Path("/kaggle/working/data.yaml").write_text("""
path: /kaggle/input/YOUR-DATASET   # the folder that holds images/ and labels/
train: images/train
val: images/val
test: images/test                  # optional
names:
  0: class_a                       # your classes, in label-index order
  1: class_b
""")
```

Train all three options with identical settings, so the backbone is the only difference:

```python
from ultralytics import YOLO

common = dict(
    data="/kaggle/working/data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    seed=0,
    deterministic=True,
    project="/kaggle/working/runs",
)
for model in ["yolo26n.pt", "yolo26n-mobilenetv4convsmall.yaml", "yolo26n-efficientnetv2b0.yaml"]:
    YOLO(model).train(name=model.rsplit(".", 1)[0], **common)
# Add "yolo26n-efficientnetv2s.yaml" or "yolo26n-mobilenetv4convmedium.yaml" to the list to also train a larger backbone.
```

Things to know on Kaggle:

- **Read-only input.** Ultralytics caches parsed labels in a `.cache` file next to the labels. Under `/kaggle/input` it cannot write one, logs a warning and re-scans the labels each run. This is harmless; copy the dataset into `/kaggle/working` if the re-scan is slow.
- **Session limits.** Interactive sessions stop when you close them or hit Kaggle's time limit. For long runs use **Save Version → Save & Run All**, which runs in the background and keeps `/kaggle/working` (including `runs/`) as the version's output.
- **Internet off.** Attach the pretrained weights as a Kaggle dataset instead and point `HF_HOME` at it (see [Pre-cache the pretrained weights](#pre-cache-the-pretrained-weights)).
- **One GPU.** The commands above use `device=0`. Multi-GPU (`device=[0, 1]`) launches DDP subprocesses and has not been tested with this fork.
- After merging this branch into `main`, change `@feat/optimized-inference` in the install line to `@main`.

---

## 6. Nano or small?

Nano. Measured in this fork, `nms=False`, FP32 LiteRT export. Latency is the median of 30 runs of the `.tflite` on a cloud x86 CPU with 4 threads — useful for **relative** comparison only, not a phone number.

| Model                          | Params | GFLOPs @640 | `.tflite` size | CPU latency @640 | CPU latency @320 |
| ------------------------------ | ------ | ----------- | -------------- | ---------------- | ---------------- |
| `yolo26n` (stock)              | 2.57M  | 6.2         | 9.9 MB         | 22.6 ms          | 6.5 ms           |
| `yolo26s` (stock)              | 10.01M | 23.1        | 38.2 MB        | 66.7 ms          | 18.7 ms          |
| `yolo26n-mobilenetv4convsmall` | 3.66M  | 7.1         | 14.2 MB        | 22.0 ms          | 6.8 ms           |
| `yolo26s-mobilenetv4convsmall` | 8.20M  | 15.6        | 31.0 MB        | 43.5 ms          | 12.0 ms          |
| `yolo26n-efficientnetv2b0`     | 7.18M  | 15.1        | 28.2 MB        | 51.2 ms          | 13.9 ms          |
| `yolo26s-efficientnetv2b0`     | 11.32M | 23.3        | 43.4 MB        | 70.2 ms          | 20.0 ms          |
| `yolo26n-efficientnetv2s`      | 21.47M | 50.4        | 85.2 MB        | 138.4 ms         | 37.8 ms          |
| `yolo26s-efficientnetv2s`      | 25.65M | 58.6        | 100.5 MB       | 155.2 ms         | 43.0 ms          |

Small costs, relative to nano:

- **Stock:** 3.9× the parameters, about 3× the latency.
- **MobileNetV4:** 2.2× the parameters, about 2× the latency.
- **EfficientNetV2-B0:** 1.6× the parameters, 1.4× the latency. The gap is smaller because the backbone, which the scale letter does not change, is most of this model.
- **EfficientNetV2-S:** 1.2× the parameters, 1.1× the latency, because the backbone dominates even more.

A bigger backbone costs more than a bigger head: `yolo26n-efficientnetv2s` is 2.7× the latency and 3× the file size of `yolo26n-efficientnetv2b0`, while `yolo26s-efficientnetv2b0` is 1.4×.

That is a large price on every option, and holding the head at nano means the backbone is the only variable between the three models. Move to `s` only if nano's accuracy is clearly short on your data — the same YAML files build it.

Also note that MobileNetV4-Conv-Small runs at essentially stock-nano speed while bringing ImageNet pretraining into the backbone.

---

## 7. How the backbone swap works

An Ultralytics model is a flat list of layers built from a YAML file. Each entry is `[from, repeats, module, args]`, and `parse_model()` in `ultralytics/nn/tasks.py` walks that list, instantiating one module per line while tracking the output channel count of every layer in a list called `ch`. A later layer refers to earlier ones by index through its `from` field.

The detection head needs **three** feature maps at strides 8, 16 and 32 (P3, P4, P5). The stock YOLO backbone produces them as three separate layers, so the neck can just reference layer indices. A classification network is a single trunk, so it does not fit that shape directly.

This fork resolves that with two pieces:

1. **A `MultiScaleBackbone` is one YAML layer that returns a _list_ of tensors** — `[P3, P4, P5]`. `parse_model()` recognizes any subclass, builds it once, and stores its `out_channels` **list** as that layer's entry in `ch`.
2. **`FeatureSelect` splits the list back into single tensors.** Three `FeatureSelect` layers read layer 0 and pick index 0, 1 and 2. `parse_model()` indexes into the stored channel list so the neck gets the right input channels automatically.

From there the YAML is ordinary YOLO26: `SPPF` and `C2PSA` on P5, then the standard PAN neck and `Detect` head. Here is `yolo26n-mobilenetv4convsmall.yaml` as printed by the trainer:

```
                   from  n    params  module                                       arguments
  0                  -1  1   1261664  block.TimmBackbone                           ['mobilenetv4_conv_small', True]
  1                   0  1         0  block.FeatureSelect                          [0]              <- P3/8,  64 ch
  2                   0  1         0  block.FeatureSelect                          [1]              <- P4/16, 96 ch
  3                   0  1         0  block.FeatureSelect                          [2]              <- P5/32, 960 ch
  4                  -1  1    953792  block.SPPF                                   [960, 256, 5, 3, True]
  5                  -1  1    249728  block.C2PSA                                  [256, 256, 1]
  6                  -1  1         0  nn.Upsample                                  [None, 2, 'nearest']
  7             [-1, 2]  1         0  conv.Concat                                  [1]
  8                  -1  1    115712  block.C3k2                                   [352, 128, 1, True]
  9                  -1  1         0  nn.Upsample                                  [None, 2, 'nearest']
 10             [-1, 1]  1         0  conv.Concat                                  [1]
 11                  -1  1     30208  block.C3k2                                   [192, 64, 1, True]     # P3/8-small
 12                  -1  1     36992  conv.Conv                                    [64, 64, 3, 2]
 13             [-1, 8]  1         0  conv.Concat                                  [1]
 14                  -1  1     95232  block.C3k2                                   [192, 128, 1, True]    # P4/16-medium
 15                  -1  1    147712  conv.Conv                                    [128, 128, 3, 2]
 16             [-1, 5]  1         0  conv.Concat                                  [1]
 17                  -1  1    463104  block.C3k2                                   [384, 256, 1, True, 0.5, True]  # P5/32-large
 18        [11, 14, 17]  1    309656  head.Detect                                  [80, 1, True, [64, 128, 256]]
Model summary: 403 layers, 3,663,800 parameters, 7.1 GFLOPs
```

The neck never sees channel numbers you had to write down — `352 = 256 (C2PSA) + 96 (P4)` and `192 = 128 (layer 8) + 64 (P3)` fall out of the backbone's reported `out_channels`. Swap the timm model name and every downstream channel count follows.

One thing the table shows: MobileNetV4's P5 output is 960 channels wide, so `SPPF` alone (954K parameters) is nearly as large as the whole backbone (1.26M).

### The `parse_model()` changes

Three small edits in `ultralytics/nn/tasks.py` make this work:

```python
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m_ = None  # pre-built module, set by branches that must instantiate early (i.e. MultiScaleBackbone)
```

```python
        elif isinstance(m, type) and issubclass(m, MultiScaleBackbone):
            m_ = m(*args)  # build here so the encoder and its pretrained weights are only loaded once
            c2 = m_.out_channels  # list of per-scale channels, i.e. [P3, P4, P5], kept as one entry in 'ch'
        elif m is FeatureSelect:
            c2 = ch[f][args[0]]  # index into the backbone's channel list, i.e. 2 -> P5 channels
```

```python
        if m_ is None:
            m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
```

The `m_ = None` slot matters. A backbone has to be instantiated inside the branch to read its `out_channels`, but the generic construction line below would otherwise build it a **second** time — downloading and initializing the pretrained encoder twice on every model load. Guarding that line with `if m_ is None` reuses the instance the branch already made.

Because the branch tests `issubclass(m, MultiScaleBackbone)` rather than a hard-coded list of classes, **new extractors need no further edits to `parse_model()`**.

---

## 8. Adding another feature extractor

### Another timm model: YAML only

Any timm model that supports `features_only=True` works through `TimmBackbone` without code changes. Copy one of the two configs, change the model name in layer 0, and name the file after it — the timm name without the `tf_` prefix and underscores:

```yaml
- [-1, 1, TimmBackbone, [mobilenetv4_hybrid_medium, True]] # [timm model, pretrained] -> yolo26-mobilenetv4hybridmedium.yaml
```

Measured candidates from the same two families (`features_only`, 3 scales, ImageNet weights):

| timm model                                   | Backbone params | `out_channels` (P3, P4, P5) | Full `n` model params | GFLOPs @640 |
| -------------------------------------------- | --------------- | --------------------------- | --------------------- | ----------- |
| `mobilenetv4_conv_small`                     | 1.26M           | `[64, 96, 960]`             | 3.66M                 | 7.1         |
| `mobilenetv4_conv_medium`                    | 7.20M           | `[80, 160, 960]`            | 9.61M                 | 17.9        |
| `mobilenetv4_hybrid_medium` (adds attention) | 8.56M           | `[80, 160, 960]`            | 10.97M                | 19.8        |
| `tf_efficientnetv2_b0`                       | 5.61M           | `[48, 112, 192]`            | 7.18M                 | 15.1        |
| `tf_efficientnetv2_b1`                       | 6.61M           | `[48, 112, 192]`            | 8.18M                 | 20.1        |
| `tf_efficientnetv2_b3`                       | 12.46M          | `[56, 136, 232]`            | 14.06M                | 29.9        |
| `tf_efficientnetv2_s`                        | 19.85M          | `[64, 160, 256]`            | 21.47M                | 50.4        |

All of them have strides 8, 16 and 32.

Rules for the YAML:

- **Keep the `scales:` block.** Without it `depth` and `width` fall back to `1.0` and the neck is built at full width — the MobileNetV4 config balloons from 3.7M to 29.1M parameters. A second, subtler effect: `parse_model` tests `if scale in "mlx"` to force `C3k` blocks, and an empty scale string is a substring of `"mlx"`, so every `C3k2` you declared with `c3k=False` is silently promoted.
- **Keep `end2end: True` and `reg_max: 1`.** They are what make the head YOLO26 (see [section 1](#check-that-a-checkpoint-really-is-yolo26)).
- Each `FeatureSelect` reads from layer `0`, not `-1`.
- The head references P3/P4/P5 by the `FeatureSelect` indices (`1`, `2`) and the post-`C2PSA` index (`5`). If you add or remove layers, renumber every `from` field and the final `Detect` line.
- Name the file without a scale letter and load it with one, as in [section 1](#1-the-three-model-options).

### An encoder outside timm: write a class

Only needed for a non-standard forward, extra adapters, or an encoder timm does not have.

**Step 1 — write the module in `ultralytics/nn/modules/block.py`** and add its name to `__all__`:

```python
class MyBackbone(MultiScaleBackbone):
    """One-line summary of the extractor."""

    def __init__(self, model: str = "my_encoder", pretrained: bool = True):
        import my_library  # scope the import so it stays an optional dependency

        super().__init__()  # call first: it resets out_channels to []
        self.m = my_library.create(model, pretrained=pretrained)
        self.out_channels = [64, 160, 256]  # channels of P3, P4, P5
        self.out_strides = [8, 16, 32]  # reduction of P3, P4, P5

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.m(x)[-3:]  # must be [P3, P4, P5], fine to coarse
```

The contract: set `out_channels` fine-to-coarse, return a list of exactly that many tensors in the same order, make the strides exactly 8/16/32 (the neck's `Concat` fails otherwise), and import third-party libraries inside `__init__` — `block.py` is imported by `import ultralytics`, so a top-level import makes the library a hard dependency of the whole package.

**Step 2 — export it** from `ultralytics/nn/modules/__init__.py`, in both the `from .block import (...)` list and `__all__`.

**Step 3 — import it in `ultralytics/nn/tasks.py`:**

```python
from ultralytics.nn.modules import (
    ...,
    MyBackbone,  # noqa: F401 - resolved by name from globals() in parse_model()
)
```

This step is easy to skip and the failure is confusing. `parse_model()` turns the module _string_ from the YAML into a class with `globals()[m]`, looking in `tasks.py`'s own namespace, so a class that is not imported there raises `KeyError: 'MyBackbone'`.

**Step 4 — write the YAML** following the rules above, with `MyBackbone` in layer 0, and train it the usual way. Check the printed layer table before letting a long run start; if the channel counts feeding `Concat` look wrong, the problem is in `out_channels`, not in the neck.

---

## 9. Offline and low-resource deployment

### Pre-cache the pretrained weights

`pretrained=True` downloads the backbone from the Hugging Face Hub on first use (`mobilenetv4_conv_small.e2400_r224_in1k`, `tf_efficientnetv2_b0.in1k`, `tf_efficientnetv2_s.in21k_ft_in1k`, `mobilenetv4_conv_medium.e500_r256_in1k`). On a machine with no outbound access, warm the cache somewhere with a network, copy it across, and pin its location:

```bash
export HF_HOME=/opt/model-cache/huggingface  # where timm weights are cached
export HF_HUB_OFFLINE=1                      # fail loudly instead of hanging on a download
```

To warm the cache:

```python
from ultralytics.nn.modules import TimmBackbone

TimmBackbone("mobilenetv4_conv_small", pretrained=True)
TimmBackbone("tf_efficientnetv2_b0", pretrained=True)
TimmBackbone("tf_efficientnetv2_s", pretrained=True)
TimmBackbone("mobilenetv4_conv_medium", pretrained=True)
```

If no pretrained weights are available at all, set the second argument in layer 0 to `False` and expect to train considerably longer.

### Freeze the extractor

The whole feature extractor is layer 0, so freezing it is one argument:

```bash
yolo detect train model=yolo26n-mobilenetv4convsmall.yaml data=my_data.yaml freeze=1
```

For `yolo26n-mobilenetv4convsmall` that freezes the 1,261,664 backbone parameters and trains the 2,402,136 in the neck and head. Ultralytics holds frozen BatchNorm layers in eval mode, so the ImageNet statistics survive small-batch training. Freeze for a short warm-up on a small dataset or when GPU memory is tight; unfreeze for the final run if you can afford it.

### Export for Android phones (LiteRT)

```bash
# FP32; the GPU delegate runs it in FP16 on the phone
yolo export model=runs/detect/train/weights/best.pt format=litert nms=False imgsz=640
# INT8 weights, FP32 activations; no calibration data needed
yolo export model=runs/detect/train/weights/best.pt format=litert nms=False imgsz=640 quantize=w8a32
# full static INT8 (weights and activations), calibrated on your data
yolo export model=runs/detect/train/weights/best.pt format=litert imgsz=640 quantize=8 data=my_data.yaml
```

`nms=False` selects the YOLO26 NMS-free head: the output is `(1, 300, 6)`, rows of `x1, y1, x2, y2, score, class`, with no post-processing to write on the phone. The default (`nms=None`) exports raw predictions, `(1, 4 + nc, 8400)` at 640, which need NMS on the device.

What each option gives you, measured on the two backbone configs at `imgsz=320`:

| `quantize`  | Output                 | MobileNetV4 `.tflite` | EfficientNetV2-B0 `.tflite` | Notes                                                                |
| ----------- | ---------------------- | --------------------- | --------------------------- | -------------------------------------------------------------------- |
| none (FP32) | `(1, 300, 6)` NMS-free | 14.3 MB               | 28.4 MB                     | Best accuracy; the GPU delegate runs it in FP16                      |
| `w8a32`     | `(1, 300, 6)` NMS-free | 4.1 MB                | 8.2 MB                      | About 3.5× smaller; activations stay FP32, so compute is not integer |
| `8`         | `(1, 4 + nc, N)` raw   | 4.3 MB                | 8.8 MB                      | Fully integer, for CPU/NPU speed; **NMS must run on the phone**      |

Upstream Ultralytics switches off the NMS-free head for static INT8 LiteRT exports (`quantize=8` and `w8a16`) and logs `LiteRT INT8 export does not support end2end models`, because static quantization breaks its class-index output. `nms=False` is ignored there, so budget for an NMS step in the app if you ship that file. Calibration wants 300+ representative images from your own data.

- **`format=litert`** replaces `format=tflite`, which still works but logs a deprecation warning. The output is a normal `.tflite` file.
- **Validate after quantizing.** INT8 can cost accuracy, and inverted-residual blocks are sensitive to it. `yolo val model=best_int8.tflite data=my_data.yaml` runs the exported file (including NMS for the raw output); compare against the `.pt` before shipping.
- **Measure on real phones.** Use LiteRT's `benchmark_model` tool on a low-end and a mid-range device, with the CPU (XNNPACK) and GPU delegates. NNAPI is deprecated as of Android 15. Depthwise-heavy MobileNets can behave differently from what GFLOPs suggest.
- **Export at the size you trained at.** Dropping `imgsz` to 416 or 320 cuts latency roughly quadratically (see [section 6](#6-nano-or-small)), but small objects in high-resolution frames suffer, so validate at that size first.

### Laptops with a 6 GB GPU

All three nano models are far below what a 6 GB card holds for inference. Use TensorRT FP16 (`format=engine quantize=16`) or ONNX Runtime with CUDA (`format=onnx`), or run the `.pt` directly with `quantize=16` for FP16 compute. These paths are standard Ultralytics exports but were not tested in this fork, because the test machine has no GPU.

---

## 10. Troubleshooting

| Symptom                                                                             | Cause and fix                                                                                                                                                                     |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model has `reg_max` 16 and no `one2one_cv2` — it is a YOLOv8/YOLO11-style head      | The YAML lacks `end2end: True` / `reg_max: 1`, or training started from a `yolov8*.pt`. Use the configs in this repo; see [section 1](#check-that-a-checkpoint-really-is-yolo26). |
| `FileNotFoundError` for `yolo26n-mobilenetv4.yaml` or `yolo26n-efficientnetv2.yaml` | The configs were renamed to `yolo26-mobilenetv4convsmall.yaml` and `yolo26-efficientnetv2b0.yaml`.                                                                                |
| `KeyError: 'MyBackbone'` in `parse_model`                                           | The class is not imported in `ultralytics/nn/tasks.py`. See [Step 3](#an-encoder-outside-timm-write-a-class).                                                                     |
| `ModuleNotFoundError: No module named 'timm'`                                       | Install the extra: `pip install -e ".[backbones]"`.                                                                                                                               |
| `Sizes of tensors must match` inside `Concat`                                       | The extractor's strides are not 8/16/32, or `out_channels` does not match what `forward` returns. Print `backbone.out_strides` and the real output shapes.                        |
| `TypeError: 'int' object is not subscriptable` at a `FeatureSelect`                 | Its `from` field points at something other than the backbone layer. All three must read from layer `0`.                                                                           |
| `IndexError: list index out of range` at a `FeatureSelect`                          | `out_channels` is empty, usually because `super().__init__()` was called _after_ assigning it and reset it to `[]`. Call it first.                                                |
| Model is far larger than expected                                                   | The config is missing its `scales:` block, or it was loaded without a scale letter. See [section 8](#another-timm-model-yaml-only).                                               |
| Pretrained weights download hangs or 403s                                           | No outbound access to the Hugging Face Hub. Pre-cache the weights or set `pretrained` to `False`.                                                                                 |
| `format='tflite' is deprecated` warning                                             | Use `format=litert`; the output is still a `.tflite` file.                                                                                                                        |
| INT8 `.tflite` outputs `(1, 4 + nc, N)` even with `nms=False`                       | Expected for static INT8 (`quantize=8`/`w8a16`); run NMS in the app, or export `quantize=w8a32` to keep the NMS-free head. See [section 9](#export-for-android-phones-litert).    |

Verified in this fork for both `yolo26n-mobilenetv4convsmall.yaml` and `yolo26n-efficientnetv2b0.yaml`: model build with the YOLO26 head (`reg_max` 1, NMS-free branch present), 2-epoch CPU training on `coco8`, validation, and LiteRT export in FP32, `w8a32` and static INT8, each loaded and run with the LiteRT interpreter, plus `yolo val` on the INT8 file. These checks ran with `pretrained=False`, since the test machine had no route to the Hugging Face Hub, so they prove the pipeline works, not the accuracy.

---

## 11. Other fork changes

`Annotator.box_label` in `ultralytics/utils/plotting.py` normalizes corner order before drawing:

```python
if not isinstance(box[0], list):  # only for standard boxes, not multi_points
    box = [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
```

Without it, a box whose `x1 > x2` or `y1 > y2` — which some augmentation and de-scaling paths can produce — raises inside PIL when the label background rectangle is drawn. This is unrelated to the backbone work but keeps plotting during training reliable.

---

## License

This fork inherits the **AGPL-3.0** license from upstream Ultralytics. See [LICENSE](LICENSE).

AGPL-3.0 is a strong copyleft license: if you run a modified version of this software as a network service, you must make the complete corresponding source available to its users. For commercial use without those obligations, request an Enterprise License from [Ultralytics Licensing](https://www.ultralytics.com/license).

Upstream project: <https://github.com/ultralytics/ultralytics> · Documentation: <https://docs.ultralytics.com>
