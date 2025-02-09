import os
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

home_dir = os.path.expanduser("~")
train_path = os.path.join(home_dir, "OneDrive", "デスクトップ", "機械学習", "signate", "PCゲーム勝敗予想", "csv",  "train.csv")
train = pd.read_csv(train_path)

# 説明変数と予測変数の決定
X = train[["blueFirstBlood", "blueKills", "blueDeaths", "blueAssists", "blueEliteMonsters", "blueDragons", "blueTotalGold", "blueTotalExperience"]]

Y = train["blueWins"]

# 説明変数の標準化
X_scaled = StandardScaler().fit_transform(X)

# モデルの学習
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, random_state=66, test_size=0.3)
hgb = HistGradientBoostingClassifier(
    loss='log_loss',                # 損失関数（'log_loss'がデフォルト）
    learning_rate=0.1,             # 学習率
    max_iter=100,                  # イテレーション数（弱学習器の数）
    max_leaf_nodes=31,             # 決定木の最大葉ノード数
    max_depth=None,                # 決定木の最大深さ（Noneで制限なし）
    min_samples_leaf=20,           # 各葉ノードに必要な最小サンプル数
    l2_regularization=0.0,         # L2正則化の強さ
    max_bins=255,                  # ビン化の最大数
    categorical_features=None,     # カテゴリ型特徴量の指定（Noneで自動検出）
    monotonic_cst=None,            # 特徴量の単調性制約
    warm_start=False,              # 既存モデルからの学習継続
    early_stopping=True,           # 早期停止の有効化
    validation_fraction=0.1,       # 検証データの割合
    n_iter_no_change=10,           # 早期停止の判定回数
    tol=1e-7,                      # 早期停止の改善許容誤差
    scoring=None,                  # モデル評価指標（デフォルトは精度）
    random_state=42,               # 乱数シード
)
hgb.fit(X_train, y_train)

# 精度の評価
print("Training set score {:.5f}".format(hgb.score(X_train, y_train)))
print("Test set score {:.5f}".format(hgb.score(X_test, y_test)))
