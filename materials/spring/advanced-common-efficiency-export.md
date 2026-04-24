# 発展テーマ候補：共通基盤の効率化と export

- 対象: B4・M
- 種別: 前期 11〜13 回の年度別発展テーマ候補
- 運用: このテーマを選んだ年度は、前期 11〜13 回の 3 回を使って `CPU profiling` と `TorchScript export` を扱う。画像・音響・共通基盤を同じ年度にすべて扱う前提にはしない。

## この回の目標
- baseline 推論時間を測定し、平均値を記録できる。
- `torch.jit.trace` で TorchScript 化して保存できる。
- batch size が推論時間の見え方にどう影響するかを説明できる。

## ミニ講義
- profiling では 1 回だけ測るとばらつきが大きい。warm-up の後で複数回測り、平均を取ると比較しやすい。
- export は保存形式の変換だけではなく、「その model が指定した入力形状で辿れるか」を確認する作業でもある。trace 前後で shape を比べるのは最低限の整合性確認になる。
- 推論時間は batch size によって見え方が変わる。1 回あたり時間と 1 サンプルあたり時間は別物なので、どちらを比較しているか意識する必要がある。

## 演習
### 基礎レベル
1. `advanced_common_efficiency_export.py` を作成し、10 次元 logits を返す小さな model と batch size 64 の固定入力を用意して、warm-up 後に複数回の推論時間を測定する。
2. batch size を 1 と 64 で比較し、1 回あたり時間と 1 サンプルあたり時間を分けて計算する。
3. `torch.jit.trace` で model を TorchScript 化し、`advanced_common_traced_model.pt` として保存する。trace 前後で出力 shape が一致することを確認する。
4. `advanced_common_profile_summary.txt` と `advanced_common_efficiency_report.md` を作成し、測定条件、平均時間、batch size による見え方の違い、trace 前後の確認結果を書く。

### 発展レベル
1. 測定回数を変えた場合に平均値がどれくらい揺れるかを確認する。
2. report に `## 測定の不確実性` を追加し、1 回だけの時間を比較根拠にしない理由と、今回の測定でまだ足りない条件を説明する。

## 確認ポイント
- `advanced_common_profile_summary.txt` が作成されている。
- `advanced_common_traced_model.pt` が作成されている。
- trace 前後の出力 shape がどちらも `(64, 10)` である。
- report に、batch size による見え方の違いが説明されている。

## 詰まったときに見る資料
- PyTorch docs: [`torch.jit.trace`](https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html)
- Python docs: [`time.perf_counter`](https://docs.python.org/3/library/time.html#time.perf_counter)
