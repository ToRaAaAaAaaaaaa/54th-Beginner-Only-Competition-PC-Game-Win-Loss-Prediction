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
    loss='log_loss', 
    max_iter=100, 
    max_depth=None, 
    max_leaf_nodes=31, 
    min_samples_leaf=3, 
    learning_rate=0.01, 
    random_state=42
    )
hgb.fit(X_train, y_train)

# 精度の評価
print("Training set score {:.5f}".format(hgb.score(X_train, y_train)))
print("Test set score {:.5f}".format(hgb.score(X_test, y_test)))
