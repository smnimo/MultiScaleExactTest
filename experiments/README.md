# experiments

MSET（`mset` パッケージ）を使った実験スクリプト。どれもこのディレクトリから実行する（`cd experiments && python main_toy.py`）。
データは `../data/` から読む。git で管理しているデータは `data/gazedata.csv` と `data/behavior/` だけで、
レビューのデータ（Amazon, Yelp）は各自で置く。セットアップはリポジトリ直下の README.md を参照。

| スクリプト | データ | 何を確かめるか | 論文 |
|---|---|---|---|
| `main_toy.py` | 乱数で作る toy 系列（2 本） | 正解の分かっている系列で、MSET が際立つ期間を検出できるか。One-class SVM・尤度と比べる | 使用（toy example） |
| `main_toy_1000.py` | toy 系列を 1000 回作り直す | `main_toy.py` の比較を多数回くり返した平均の精度 | 使用（精度の表） |
| `main_gaze.py` | `data/gazedata.csv` | 実際の視線データで、どの属性値に注目していたか（attributes-of-focus） | 使用（gaze data） |
| `main_review_cat.py` | Amazon / Yelp レビュー | 1 商品の星評価の系列で、評価が偏った期間（categorical） | 未使用 |
| `main_review_ord.py` | Amazon / Yelp レビュー | 同上（ordinal） | 未使用 |
| `main_review_ord_words.py` | Amazon / Yelp レビュー | 同上（ordinal）と、偏った期間のレビューに多い単語 | 未使用 |
| `main_cat_treetagger_each.py`, `main_cat_treetagger_all.py` | Amazon レビュー、TreeTagger | 否定的・肯定的な評価が集中した期間のレビューを特徴づける単語（categorical、2 値） | 未使用 |
| `main_ord_treetagger_each.py`, `main_ord_treetagger_all.py` | Amazon レビュー、TreeTagger | 際立つ期間のレビューを特徴づける単語（ordinal） | 未使用 |
| `behavior/main_behavior.py`, `behavior/makedatelist.py` | `data/behavior/` | MSET 以前の、単一スケールの exact test の試行 | 未使用 |

## toy 系列（`main_toy.py`, `main_toy_100.py`）

**系列の作り方**：値は 0〜3 の 4 種類。中立の区間（20 個ずつ 4 つ）と際立つ区間（10, 5, 8 個）が交互に並ぶ長さ 103 の系列。

- 中立の区間は中立分布から引く。系列 1 は `[0.25, 0.25, 0.25, 0.25]`、系列 2 は `[0.3, 0.25, 0.25, 0.2]`。
- 際立つ区間は `[0.9, 0.025, 0.05, 0.025]` から引く（ほとんど値 0）。際立つ区間が正解（`gtflag`）。

**比べる方法**：どれも検出した時点を正解と比べ、precision・recall・accuracy（`main_toy_100.py` は F 値も）を出す。

- MSET：`MSET_cat`（窓サイズ 1〜10）、p < 0.01 の窓の和集合。
- One-class SVM：窓サイズ 5, 10 の度数ベクトルで、中立分布から引いた 1000 個で学習し、外れ値と判定された窓を検出とする。
- 尤度（LD）：窓サイズ 5, 10 の度数の多項分布の確率がしきい値（0.01, 0.001, 0.0001）未満の窓を検出とする。

**`main_toy.py`**：seed 4 の 1 回分。系列 1・2 それぞれについて、度数のヒストグラム、p 値と際立つ期間の図、
One-class SVM と尤度の検出結果の図を出し、精度を表示する。

**`main_toy_1000.py`**：seed 0 で 1000 回系列を作り直し、各方法の精度の平均を表示する。
図は出さない。

## 視線データ（`main_gaze.py`）

`data/gazedata.csv` は 5 行で 1 組（5 属性）。各行は、注視した商品のその属性の値（1〜3）の系列。
中立分布を一様（3 値とも 1/3）として、16 番目の組（`u = 16`）の 5 属性それぞれに `MSET_cat` をかけ、p 値と際立つ期間・値の図を出す。
際立つ値が、その時点で注目していた属性値（attributes-of-focus）にあたる。値は 1 始まりなので `- 1` して渡している。

## レビュー（`main_review_*.py`）

1 商品のレビューを日付順に並べた星評価（1〜5）の系列に MSET をかけ、評価が偏った期間を探す。
中立分布は、その商品の全期間の星評価の割合。

- `main_review_cat.py`：`MSET_cat`。p 値と際立つ期間・値の図を出し、際立つ期間のレビューの日付・星・本文を表示する。
- `main_review_ord.py`：`MSET_ord`（星評価を順序のある値として、区切りごとに検定）。表示は `main_review_cat.py` と同じ。
- `main_review_ord_words.py`：`MSET_ord` で際立つ期間を求め、期間内と期間外で、否定的（星 2 以下）なレビューの単語の頻度分布を比べる。
  旧 `main_amazon_rev.py` の設定（データファイルとしきい値）はコメントに残してある。

`dataflag` で Amazon（`'amzn'`）と Yelp（`'yelp'`）を切り替える。Amazon は、レビューがしきい値（`thr_num`）より多い商品のうち
`i` 番目を使う。Yelp は 2017 年 1〜3 月のレビュー全体を 1 つの系列として扱う。

## レビューの単語（`main_*_treetagger_*.py`）

際立つ期間のレビューと、それ以外のレビューを分ける単語を探す。単語は TreeTagger で品詞を付け、名詞（`hinshi='meishi'`）または
名詞と動詞（`'meishi_doshi'`）の原形を使う。単語の出現回数のベクトルで ExtraTreesClassifier を学習し、
重要度の高い単語と、その単語を含むレビューを `results/amazon_reviews_us_Camera_*.txt` に書き出す。

- `main_cat_treetagger_*.py`：星評価を 2 値に分けて `MSET_cat` をかける。否定側は「星 2 以下」と「3 以上」、肯定側は「星 4 以下」と「5」。
  否定的な評価が多い期間の、星 2 以下のレビューを期間外のものと比べる。
- `main_ord_treetagger_*.py`：`MSET_ord` で際立つ期間を求め、期間内と期間外のレビューを、肯定的（星 4 以上）・否定的（星 2 以下）に分けて比べる。
- `_each`：レビューが 3000 件より多い商品ごとに 1 ファイル。`_all`：しきい値（3000, 2000, 1000）ごとに、該当する商品をまとめて 1 ファイル。

**要確認**：`main_cat_treetagger_*.py` の肯定側は、星 5 が多い期間から `dseq_pos==0`（星 4 以下）のレビューを取り出している。
否定側（星 2 以下が多い期間から星 2 以下を取り出す）と揃えるなら `dseq_pos==1`（星 5）のはず。データが無く動作を確認できないので、直していない。

## results/

| ファイル | 内容 |
|---|---|
| `results.csv` | `main_toy.py` の系列 1 の精度をまとめたもの（MSET、One-class SVM、尤度） |
| `result_100.csv`, `result_1000.csv`, `result_100_4probs_seed0.csv` | `main_toy_100.py` の出力を保存したもの（試行回数や中立分布の数が違う版。2019 年当時のコードの出力） |
| `toy_data_*.pdf`, `MSET_toy_*.pdf`, `1svm_*.pdf`, `likelihood_*.pdf` | `main_toy.py` の図（toy 系列、MSET、One-class SVM、尤度） |

`results.csv` の MSET と尤度（窓 5、しきい値 0.01）の値は、今の `main_toy.py` の系列 1 の出力と一致する。
One-class SVM は一致しない（学習データの乱数の並びなど、当時と条件が違うとみられる）。

## 注意

- TreeTagger を使うスクリプトは TreeTagger 本体が必要。インストール先は各スクリプトの `TAGDIR` に直接書かれているので、環境に合わせて書き換える。
- レビューのスクリプトは、当時のデータが既に手元に無いため、整理の後に実行を確認していない（構文の確認だけ）。
