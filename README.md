# Ultralytics YOLO with Custom Feature Extractors

A fork of [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) that lets you replace the YOLO backbone with a pretrained image classifier — MobileNet, EfficientNetV2, ResNet50, or anything else in [timm](https://github.com/huggingface/pytorch-image-models) or [torchvision](https://pytorch.org/vision/stable/models.html) — while keeping the stock YOLO neck and detection head.

The target is object detection that runs **offline on a tight compute budget**: a YOLO26 head on top of a MobileNetV3 or EfficientNetV2 feature extractor. Everything else in Ultralytics — training loop, augmentation, validation, export — works unchanged.

- **Upstream sync:** Ultralytics **v8.4.126**
- **License:** AGPL-3.0, same as upstream (see [License](#license))
- **Full Ultralytics documentation:** <https://docs.ultralytics.com>

---

## Contents

1. [What this fork adds](#1-what-this-fork-adds)
2. [Install](#2-install)
3. [Quick start](#3-quick-start)
4. [How the backbone swap works](#4-how-the-backbone-swap-works)
5. [Adding your own feature extractor](#5-adding-your-own-feature-extractor)
6. [Choosing a backbone](#6-choosing-a-backbone)
7. [Offline and low-resource deployment](#7-offline-and-low-resource-deployment)
8. [Troubleshooting](#8-troubleshooting)
9. [Other fork changes](#9-other-fork-changes)
10. [License](#license)

---

## 1. What this fork adds

| File | Change |
| --- | --- |
| `ultralytics/nn/modules/block.py` | `MultiScaleBackbone` base class, plus `TimmBackbone`, `TorchvisionBackbone`, `MobileNetV3Backbone`, `EfficientNetV2Backbone`, `ResNet50Backbone` and `FeatureSelect` |
| `ultralytics/nn/modules/__init__.py` | Re-exports the new modules from the package |
| `ultralytics/nn/tasks.py` | Imports the backbones and teaches `parse_model()` to handle a layer that returns *several* feature maps |
| `ultralytics/cfg/models/26/yolo26-mobilenetv3.yaml` | Ready-to-train YOLO26 + MobileNetV3 config |
| `ultralytics/cfg/models/26/yolo26-efficientnetv2.yaml` | Ready-to-train YOLO26 + EfficientNetV2-S config |
| `ultralytics/cfg/models/26/yolo26-resnet50.yaml` | ResNet50 config, kept as the heavyweight accuracy baseline |
| `ultralytics/utils/plotting.py` | Normalizes box corner order in `Annotator.box_label` (see [Other fork changes](#9-other-fork-changes)) |
| `pyproject.toml` | `backbones` optional extra that pulls in `timm` |

Nothing else in the library was modified, so upstream releases can still be merged in.

---

## 2. Install

```bash
git clone https://github.com/kazuretsu/ultralytics.git
cd ultralytics
pip install -e ".[backbones]"     # editable install + timm
```

`timm` is **optional**. It is imported lazily inside the backbone constructors, so `import ultralytics` and every stock YOLO model still work without it. You only need it for `TimmBackbone`, `MobileNetV3Backbone` and `EfficientNetV2Backbone`. `TorchvisionBackbone` and `ResNet50Backbone` have no dependency beyond torchvision, which Ultralytics already requires.

---

## 3. Quick start

```python
from ultralytics import YOLO

model = YOLO("yolo26n-mobilenetv3.yaml")           # or yolo26n-efficientnetv2.yaml
model.train(data="my_data.yaml", epochs=100, imgsz=640, batch=16)
model.val()
model.export(format="onnx")                        # onnx / openvino / tflite ...
```

Or from the CLI:

```bash
yolo detect train model=yolo26s-efficientnetv2.yaml data=my_data.yaml epochs=100 imgsz=640
```

**Start from a `.yaml`, never a `.pt`.** There is no pretrained YOLO checkpoint for a swapped backbone. The feature extractor starts from its ImageNet weights, and the neck and head start random — so plan for a longer schedule than fine-tuning a stock `yolo26n.pt`.

The scale letter (`n`, `s`, `m`, `l`, `x`) in the config name sizes the **neck and head only**. The backbone size is fixed by the timm model name inside the config.

---

## 4. How the backbone swap works

An Ultralytics model is a flat list of layers built from a YAML file. Each entry is `[from, repeats, module, args]`, and `parse_model()` in `ultralytics/nn/tasks.py` walks that list, instantiating one module per line while tracking the output channel count of every layer in a list called `ch`. A later layer refers to earlier ones by index through its `from` field.

The detection head needs **three** feature maps at strides 8, 16 and 32 (P3, P4, P5). The stock YOLO backbone produces them as three separate layers, so the neck can just reference layer indices. A classification network is a single sequential trunk, so it does not fit that shape directly.

This fork resolves that with two pieces:

1. **A `MultiScaleBackbone` is one YAML layer that returns a *list* of tensors** — `[P3, P4, P5]`. `parse_model()` recognizes any subclass, builds it once, and stores its `out_channels` **list** as that layer's entry in `ch`.
2. **`FeatureSelect` splits the list back into single tensors.** Three `FeatureSelect` layers read layer 0 and pick index 0, 1 and 2. `parse_model()` indexes into the stored channel list so the neck gets the right `c1` automatically.

From there the YAML is ordinary YOLO26: `SPPF` and `C2PSA` on P5, then the standard PAN neck and `Detect` head.

Here is the resulting graph, printed by the trainer for `yolo26n-mobilenetv3.yaml`:

```
                   from  n    params  module                                       arguments
  0                  -1  1   2971952  block.MobileNetV3Backbone                    [True, 'mobilenetv3_large_100']
  1                   0  1         0  block.FeatureSelect                          [0]              <- P3/8,  40 ch
  2                   0  1         0  block.FeatureSelect                          [1]              <- P4/16, 112 ch
  3                   0  1         0  block.FeatureSelect                          [2]              <- P5/32, 960 ch
  4                  -1  1    953792  block.SPPF                                   [960, 256, 5, 3, True]
  5                  -1  1    249728  block.C2PSA                                  [256, 256, 1]
  6                  -1  1         0  nn.Upsample                                  [None, 2, 'nearest']
  7             [-1, 2]  1         0  conv.Concat                                  [1]
  8                  -1  1    117760  block.C3k2                                   [368, 128, 1, True]
  9                  -1  1         0  nn.Upsample                                  [None, 2, 'nearest']
 10             [-1, 1]  1         0  conv.Concat                                  [1]
 11                  -1  1     28672  block.C3k2                                   [168, 64, 1, True]     # P3/8-small
 12                  -1  1     36992  conv.Conv                                    [64, 64, 3, 2]
 13             [-1, 8]  1         0  conv.Concat                                  [1]
 14                  -1  1     95232  block.C3k2                                   [192, 128, 1, True]    # P4/16-medium
 15                  -1  1    147712  conv.Conv                                    [128, 128, 3, 2]
 16             [-1, 5]  1         0  conv.Concat                                  [1]
 17                  -1  1    463104  block.C3k2                                   [384, 256, 1, True]    # P5/32-large
 18        [11, 14, 17]  1    309656  head.Detect                                  [80, 1, True, [64, 128, 256]]
YOLO26n-mobilenetv3 summary: 410 layers, 5,374,600 parameters, 7.8 GFLOPs
```

Note how the neck never sees channel numbers you had to write down — `368 = 256 (C2PSA) + 112 (P4)` and `168 = 128 (layer 8) + 40 (P3)` fall out of the backbone's reported `out_channels`. Swap `mobilenetv3_large_100` for another timm model and every downstream channel count follows automatically.

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

Because the branch tests `issubclass(m, MultiScaleBackbone)` rather than a hard-coded tuple of classes, **new extractors need no further edits to `parse_model()`**.

---

## 5. Adding your own feature extractor

If the encoder you want is already in timm or torchvision, you do not need to write any code — use the generic wrappers straight from a YAML:

```yaml
- [-1, 1, TimmBackbone, [convnext_tiny, True]] # [model name, pretrained]
- [-1, 1, TorchvisionBackbone, [efficientnet_v2_s, DEFAULT]] # [model name, weights]
```

`TorchvisionBackbone` traces the model's `.features` sequential, so it covers the MobileNet, EfficientNet, ConvNeXt and VGG families but rejects architectures built differently (ResNet, RegNet) with an explanatory error — reach for `TimmBackbone` or a subclass there.

Write a class only when you need custom behavior — a non-standard forward, extra adapters, or an encoder from neither library. Then follow these five steps.

### Step 1 — write the module in `ultralytics/nn/modules/block.py`

Subclass `MultiScaleBackbone` and honor its contract:

```python
class MyBackbone(MultiScaleBackbone):
    """One-line summary of the extractor."""

    def __init__(self, pretrained: bool = True, model: str = "my_encoder"):
        import my_library  # scope the import so it stays an optional dependency

        super().__init__()
        self.m = my_library.create(model, pretrained=pretrained)
        self.out_channels = [64, 160, 256]  # channels of P3, P4, P5
        self.out_strides = [8, 16, 32]      # reduction of P3, P4, P5

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.m(x)[-3:]  # must be [P3, P4, P5], fine to coarse
```

The rules:

- **Set `out_channels`** in `__init__`, ordered fine-to-coarse. `parse_model()` reads it to size the neck. Read it from the encoder's own metadata where you can (`feature_info.channels()` in timm) rather than hard-coding numbers, so swapping variants does not require editing the class.
- **Return a list of exactly `len(out_channels)` tensors**, in the same order.
- **Match the strides.** The neck concatenates upsampled P5 with P4, and upsampled P4 with P3, so the outputs must be exactly 1/8, 1/16 and 1/32 of the input. A backbone with different reductions will fail with a shape mismatch inside `Concat`.
- **Import third-party libraries inside `__init__`**, not at module scope. `block.py` is imported by `import ultralytics`, so a top-level `import timm` makes timm a hard dependency of the entire package for every user.

Then add the class name to `__all__` at the top of `block.py`.

### Step 2 — export it in `ultralytics/nn/modules/__init__.py`

Two places, both needed:

```python
from .block import (
    ...,
    MyBackbone,
)

__all__ = (
    ...,
    "MyBackbone",
)
```

### Step 3 — import it in `ultralytics/nn/tasks.py`

```python
from ultralytics.nn.modules import (
    ...,
    MyBackbone,  # noqa: F401 - resolved by name from globals() in parse_model()
)
```

This step is easy to skip and the failure is confusing. `parse_model()` turns the module *string* from the YAML into a class with `globals()[m]`, looking in `tasks.py`'s own namespace. A class that is not imported there raises `KeyError: 'MyBackbone'` even though the import works fine everywhere else.

### Step 4 — write the model YAML

Copy `ultralytics/cfg/models/26/yolo26-mobilenetv3.yaml` and change layer 0:

```yaml
backbone:
  # [from, repeats, module, args]
  - [-1, 1, MyBackbone, [True, my_encoder]] # 0-[P3, P4, P5]
  - [0, 1, FeatureSelect, [0]] # 1-P3/8
  - [0, 1, FeatureSelect, [1]] # 2-P4/16
  - [0, 1, FeatureSelect, [2]] # 3-P5/32
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 4
  - [-1, 2, C2PSA, [1024]] # 5
```

Points to get right:

- The `args` list is passed positionally to your constructor, so `[True, my_encoder]` means `MyBackbone(True, "my_encoder")`. Bare YAML strings are fine; `parse_model` resolves them.
- Each `FeatureSelect` reads from layer `0`, not `-1`. Only the first can use `-1`.
- The head references P3/P4/P5 by the `FeatureSelect` indices (`1`, `2`) and the post-`C2PSA` index (`5`). If you add or remove layers, renumber every `from` field and the final `Detect` line.
- **Keep the `scales:` block.** Without it `depth` and `width` fall back to `1.0` and the neck is built at full width — the same MobileNetV3 config balloons from 5.4M to 30.8M parameters. A second, subtler effect: `parse_model` tests `if scale in "mlx"` to force `C3k` blocks, and an empty scale string is a substring of `"mlx"`, so every `C3k2` you declared with `c3k=False` is silently promoted.
- Name the file without a scale letter (`yolo26-myencoder.yaml`) and load it *with* one (`yolo26n-myencoder.yaml`). Ultralytics strips the letter to find the file and uses it to pick the scale.

### Step 5 — train

```bash
yolo detect train model=yolo26n-myencoder.yaml data=my_data.yaml epochs=100 imgsz=640
```

Check the printed layer table before letting a long run start. If the channel counts feeding `Concat` look wrong, the problem is in `out_channels`, not in the neck.

---

## 6. Choosing a backbone

Feature extractor cost, measured in this fork (`features_only`, 3 scales, ImageNet-1k variants):

| timm model | Params | `out_channels` (P3, P4, P5) | Strides |
| --- | --- | --- | --- |
| `mobilenetv3_small_100` | 0.9M | `[24, 48, 576]` | 8, 16, 32 |
| `mobilenetv4_conv_small` | 1.3M | `[64, 96, 960]` | 8, 16, 32 |
| `mobilenetv3_large_100` | 3.0M | `[40, 112, 960]` | 8, 16, 32 |
| `mobilenetv4_conv_medium` | 7.2M | `[80, 160, 960]` | 8, 16, 32 |
| `tf_efficientnetv2_b0` | 5.6M | `[48, 112, 192]` | 8, 16, 32 |
| `tf_efficientnetv2_b3` | 12.5M | `[56, 136, 232]` | 8, 16, 32 |
| `tf_efficientnetv2_s` | 19.8M | `[64, 160, 256]` | 8, 16, 32 |

Complete detector cost at `imgsz=640`, `nc=80`:

| Config | Total params | Backbone params | GFLOPs |
| --- | --- | --- | --- |
| `yolo26n.yaml` (stock) | 2,572,280 | — | 6.2 |
| `yolo26s.yaml` (stock) | 10,009,784 | — | 23.1 |
| `yolo26n-mobilenetv3.yaml` | 5,374,600 | 2,971,952 | 7.8 |
| `yolo26s-mobilenetv3.yaml` | 9,912,040 | 2,971,952 | 16.2 |
| `yolo26n-efficientnetv2.yaml` | 21,468,392 | 19,847,248 | 50.4 |
| `yolo26s-efficientnetv2.yaml` | 25,653,064 | 19,847,248 | 58.6 |
| `yolo26n-resnet50.yaml` | 28,252,120 | 23,508,032 | 74.0 |

Reading the table:

- **MobileNetV3 is the offline/edge choice.** `yolo26n-mobilenetv3` costs about 25% more FLOPs than stock `yolo26n` and adds ImageNet pretraining on the encoder, which is what pays off when your detection dataset is small.
- **EfficientNetV2-S is the accuracy choice** and is heavy enough that you should check it against your latency budget before committing. Use `tf_efficientnetv2_b0` for a middle option — swap the model name in the config, no other change needed.
- **ResNet50 is a baseline, not a deployment target.** It is here so backbone comparisons have a familiar reference point.
- A stock `yolo26n`/`yolo26s` is often still the best FLOPs-per-mAP trade. Swap the backbone when you want the ImageNet prior, a specific export target, or a controlled comparison — not by default.

The scale letter changes only the neck: `yolo26s-mobilenetv3` doubles neck width over `yolo26n-mobilenetv3` on the same 3.0M-parameter extractor.

---

## 7. Offline and low-resource deployment

### Pre-cache the pretrained weights

`pretrained=True` downloads on first use — timm from the Hugging Face Hub, torchvision from `download.pytorch.org`. On a machine with no outbound access, warm the caches somewhere with a network and copy them across, then pin the cache locations:

```bash
export HF_HOME=/opt/model-cache/huggingface     # timm backbones
export TORCH_HOME=/opt/model-cache/torch        # torchvision backbones
export HF_HUB_OFFLINE=1                         # fail loudly instead of hanging on a download
```

To warm the cache:

```python
from ultralytics.nn.modules import TimmBackbone
TimmBackbone("mobilenetv3_large_100", pretrained=True)  # downloads once into HF_HOME
```

If no pretrained weights are available at all, set the first arg to `False` in the config and expect to train considerably longer.

### Freeze the extractor

The whole feature extractor is layer 0, so freezing it is one argument:

```bash
yolo detect train model=yolo26n-mobilenetv3.yaml data=my_data.yaml freeze=1
```

For `yolo26n-mobilenetv3` that freezes 2,971,952 backbone parameters and trains 2,402,648 in the neck and head. Ultralytics also holds frozen BatchNorm layers in eval mode, so the ImageNet statistics survive small-batch training. Freeze for a short warm-up on a small dataset or when GPU memory is tight; unfreeze for the final run if you can afford it.

### Export

Export works normally — the backbone is plain PyTorch:

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx  imgsz=640   # verified
yolo export model=runs/detect/train/weights/best.pt format=openvino int8=True
yolo export model=runs/detect/train/weights/best.pt format=tflite  int8=True
```

Two things to watch on the edge:

- **Depthwise convolutions** dominate MobileNet, and some runtimes handle them poorly. Benchmark on the actual target device rather than trusting the GFLOPs column.
- **INT8 quantization** hits inverted-residual blocks harder than plain convolutions. Validate quantized mAP before shipping; if it collapses, try `tf_efficientnetv2_b0`, which quantizes more gracefully than MobileNet's squeeze-and-excite blocks.

Also drop `imgsz` to 416 or 320 when latency matters — it scales FLOPs quadratically and usually costs less mAP than shrinking the model.

---

## 8. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `KeyError: 'MyBackbone'` in `parse_model` | The class is not imported in `ultralytics/nn/tasks.py`. See [Step 3](#step-3--import-it-in-ultralyticsnntaskspy). |
| `ModuleNotFoundError: No module named 'timm'` | Install the extra: `pip install -e ".[backbones]"`, or use `TorchvisionBackbone` instead. |
| `Sizes of tensors must match` inside `Concat` | The extractor's strides are not 8/16/32, or `out_channels` does not match what `forward` actually returns. Print `backbone.out_strides` and the real output shapes. |
| `TypeError: 'int' object is not subscriptable` at a `FeatureSelect` | Its `from` field points at something other than the backbone layer. All three must read from layer `0`. |
| Model is far larger than expected | The config is missing its `scales:` block, or it was loaded without a scale letter. See [Step 4](#step-4--write-the-model-yaml). |
| Pretrained weights download hangs or 403s | No outbound access to the Hugging Face Hub. Pre-cache the weights or set `pretrained=False`. |
| `IndexError: list index out of range` at a `FeatureSelect` | `out_channels` is empty, usually because `super().__init__()` was called *after* assigning it and reset it to `[]`. Call it first. |
| Backbone appears to initialize twice | The `if m_ is None:` guard in `parse_model` is missing. See [The `parse_model()` changes](#the-parse_model-changes). |

Verified in this fork for `yolo26n-mobilenetv3.yaml`: model build, 2-epoch CPU training on `coco8`, validation, prediction, ONNX export and ONNX Runtime inference all pass. The training check ran with `pretrained=False`, since the test machine had no route to the weight host.

---

## 9. Other fork changes

`Annotator.box_label` in `ultralytics/utils/plotting.py` now normalizes corner order before drawing:

```python
if not isinstance(box[0], list):  # only for standard boxes, not multi_points
    box = [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
```

Without it, a box whose `x1 > x2` or `y1 > y2` — which some augmentation and de-scaling paths can produce — raises inside PIL when the label background rectangle is drawn. This is unrelated to the backbone work but is required for plotting during training to be reliable.

---

## License

This fork inherits the **AGPL-3.0** license from upstream Ultralytics. See [LICENSE](LICENSE).

AGPL-3.0 is a strong copyleft license: if you run a modified version of this software as a network service, you must make the complete corresponding source available to its users. For commercial use without those obligations, request an Enterprise License from [Ultralytics Licensing](https://www.ultralytics.com/license).

Upstream project: <https://github.com/ultralytics/ultralytics> · Documentation: <https://docs.ultralytics.com>
