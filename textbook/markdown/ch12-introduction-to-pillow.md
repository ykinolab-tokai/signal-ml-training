<!-- Re-authored after migration because the original notebook content was too GUI-centric for self-study. -->

# Pillow 入門

## この章の目的

Pillow は，画像を読み込み，加工し，保存するための基本ライブラリです．

- 画像の読み込みと保存
- サイズ変更，回転，切り出し
- 色空間の変換
- 簡単なフィルタ処理
- NumPy 配列との相互変換

を学びます．

## Pillow とは何か

Pillow は PIL (Python Imaging Library) を引き継ぐ画像処理ライブラリです．
この教材では，`show()` に頼るより `save()` で結果を残す流れを基本にします．

## インストール

```bash
python -m pip install Pillow
```

```python
from PIL import Image, ImageDraw, ImageFilter
```

## 画像を開いて属性を確認する

```python
from pathlib import Path
from PIL import Image

output_dir = Path("outputs/ch12")
output_dir.mkdir(parents=True, exist_ok=True)

img = Image.open("path/to/image.jpg")
print(img.format)
print(img.mode)
print(img.size)
```

確認すべき属性は次です．

- `format`: JPEG や PNG などの形式
- `mode`: RGB, L, RGBA などの画素表現
- `size`: `(width, height)` の順のサイズ

## 結果を保存する流れを最初に作る

```python
img.save(output_dir / "original_copy.png")
```

## サイズ変更

```python
img_resized = img.resize((256, 256))
img_resized.save(output_dir / "resized_256.png")
```

## 回転と反転

```python
img_rotated = img.rotate(45, expand=True)
img_rotated.save(output_dir / "rotated_45.png")
```

```python
img_flip_lr = img.transpose(Image.FLIP_LEFT_RIGHT)
img_flip_tb = img.transpose(Image.FLIP_TOP_BOTTOM)

img_flip_lr.save(output_dir / "flip_left_right.png")
img_flip_tb.save(output_dir / "flip_top_bottom.png")
```

## 切り出し

```python
crop_box = (50, 50, 200, 200)
img_cropped = img.crop(crop_box)
img_cropped.save(output_dir / "cropped.png")
```

## 描画と注釈

```python
annotated = img.copy()
draw = ImageDraw.Draw(annotated)
draw.rectangle(((50, 50), (200, 200)), outline="red", width=3)
annotated.save(output_dir / "annotated_box.png")
```

## フィルタ処理

```python
img_blurred = img.filter(ImageFilter.BLUR)
img_blurred.save(output_dir / "blurred.png")
```

## 色空間の変換

```python
img_gray = img.convert("L")
img_hsv = img.convert("HSV")

img_gray.save(output_dir / "gray.png")
img_hsv.save(output_dir / "hsv.png")
```

## NumPy 配列との相互変換

```python
import numpy as np

img_array = np.array(img)
print(img_array.shape)
print(img_array.dtype)
```

```python
img_from_array = Image.fromarray(img_array)
img_from_array.save(output_dir / "from_array.png")
```

## 特定画素の値を調べる

```python
rgb = img.getpixel((50, 50))
print(rgb)
```

## ヒストグラム

```python
import matplotlib.pyplot as plt

gray = img.convert("L")
histogram = gray.histogram()

plt.figure(figsize=(6, 4))
plt.bar(range(256), histogram)
plt.xlabel("pixel value")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(output_dir / "gray_histogram.png", dpi=150)
plt.close()
```

## 二値化

```python
gray = img.convert("L")
threshold = 127
img_binarized = gray.point(lambda x: 0 if x < threshold else 255)
img_binarized.save(output_dir / "binarized.png")
```

## 最小の前処理パイプライン

```python
from pathlib import Path
from PIL import Image

input_path = "path/to/image.jpg"
output_dir = Path("outputs/ch12")
output_dir.mkdir(parents=True, exist_ok=True)

img = Image.open(input_path)
img = img.convert("L")
img = img.resize((128, 128))
img.save(output_dir / "preprocessed_128_gray.png")
```

## この章で押さえるべき点

- Pillow は画像ファイルを `Image` オブジェクトとして扱う基本ライブラリである
- `size`, `mode`, `format` を最初に確認すると混乱しにくい
- `show()` に頼るより `save()` で結果を残す方が教材・演習では重要である
- resize, crop, rotate, flip, filter, convert は画像前処理の基本操作である
- NumPy 配列へ変換すると，機械学習や数値処理へつなげやすい
