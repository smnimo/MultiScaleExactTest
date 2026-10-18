# MSET: Multi-Scale Exact Test

選択行動の系列から「偏った行動を行っている期間」と、その期間に偏って選ばれた値を検出する手法 MSET の実装と実験スクリプト。検出は確率分布に従った一般的な行動を行った場合に対し、「際立って異なった、偏りのある」行動区間。

> K. Shimonishi and H. Kawashima, "MSET: Multi-Scale Exact Test for Detecting Distinctive Periods in Choice Behavior,"
> AAAI 2020

窓サイズ 1〜`win_max` のすべての窓について、中立分布（多項分布）の下での exact test の p 値を計算し、
p < α の窓を「際立つ窓」とする。際立つ窓を窓・スケールをまたいで和集合にしたものが際立つ期間になる。

- categorical：窓内の度数の多項分布で検定する。特定の窓幅で、中立より割合が高い値を「際立つ値」とする。
- ordinal：区切りごとに二項分布で検定する。特定の区切りについて、窓が偏っている側（区切り以下か、区切りより上か）の水準を「際立つ値」とする。

## ディレクトリ

```
pyproject.toml
mset/                MSET 本体（パッケージ）
  core.py            p 値の表、際立つ期間と値（`import mset` で使える）
  plot.py            際立つ期間・値の計算と描画（`from mset import plot`）
tests/               テスト（scipy で書いた参照実装との照合など）
experiments/         実験スクリプトと結果
  main_toy.py        toy データの実験（図・results/results.csv）
  main_toy_100.py    toy データの実験の繰り返し（results/result_*.csv）
  main_gaze.py       視線データ（data/gazedata.csv）
  main_review_cat.py, main_review_ord.py   Amazon / Yelp レビューの 1 商品を MSET で可視化
  main_review_ord_words.py                 同上（ordinal）と、際立つ期間のレビューの単語の集計
  main_cat_*.py, main_ord_*.py             レビューの実験（TreeTagger）
  results/           実験結果
data/                データ（git 管理は gazedata.csv だけ。レビューのデータは各自で置く）
```

## セットアップ

リポジトリ直下（pyproject.toml があるところ）で実行する。

```sh
uv venv
uv pip install -e .                  # MSET 本体
uv pip install -e ".[experiments]"   # 実験スクリプトも動かす場合
```

- TreeTagger を使うスクリプト（`main_*_treetagger_*.py`）は TreeTagger 本体も必要。
  インストール先は各スクリプトの `TAGDIR` に直接書かれているので、環境に合わせて書き換える。
- 実験スクリプトは `experiments/` から実行する（データを `../data/` で参照している）。

## テスト

```sh
uv pip install --group test   # pytest（開発用の依存グループ。パッケージの依存には含まれない）
.venv/bin/python -m pytest    # リポジトリ直下で
```

## 使い方

```python
import numpy as np
import mset
from mset import plot as mset_plot   # 描画を使う場合

neu_prob = [0.25, 0.25, 0.25, 0.25]       # 中立分布（値 0..3 の確率）
seq = np.array([1, 2, 3, 0, 1, 2, 0, 0, 0, 0, 0, 0])   # 値は 0..nbins-1 の整数

mcat = mset.MSET_cat(neu_prob, win_max=10)
pvalues = np.array(mcat.calc_pvalues_allseq(seq))   # [窓サイズ-1][t]、窓が収まらない所は 2
period = mset.find_dperiod(pvalues, 0.05, 'cat')                # (T,) bool：t = 2 以降
values = mset.find_dperiod_value(seq, mcat, pvalues, 0.05, 'cat')  # (nbins, T) bool、i 行目が値 i：値 0
```

ordinal は `MSET_ord` と `'ord'` を使う。p 値は `[窓サイズ-1][t][区切り]` で、区切り thr は「thr 以下か、それより上か」を表す。

値が 0..nbins-1 の整数でないと `mset.MSET_Error` になる。1 始まりのデータ（星評価 1〜5、視線データ 1〜3 など）は
`seq - 1` を渡す。

- 有意の判定は `mset.is_significant`（p < α）。丸め誤差の範囲で α に等しい p は有意としない。
  exact test で確率を比べるときも、相対誤差 1e-9 の範囲は同率として扱う。
- `mset.plot.MSET_plot_cat` / `MSET_plot_ord` の `calc_dperiods(alpha=0.01)` は `find_dperiod_value` と同じ結果を返す。
  ordinal の `mode='legacy'` は AAAI 2020 の値の付け方（下側は、それより下の水準にまだ印が無い時点だけに付ける）。
