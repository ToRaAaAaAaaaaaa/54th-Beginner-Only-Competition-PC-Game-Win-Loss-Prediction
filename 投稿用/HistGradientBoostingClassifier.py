import os
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

home_dir = os.path.expanduser("~")
train_path = os.path.join(home_dir, "OneDrive", "デスクトップ", "機械学習", "signate", "PCゲーム勝敗予想", "csv",  "train.csv")
train = pd.read_csv(train_path)
test_path = os.path.join(home_dir, "OneDrive", "デスクトップ", "機械学習", "signate", "PCゲーム勝敗予想", "csv",  "test.csv")
test = pd.read_csv(test_path)

# 説明変数と予測変数の決定
features = train[["blueFirstBlood", "blueKills", "blueDeaths", "blueAssists", "blueEliteMonsters", "blueDragons", "blueTotalGold", "blueTotalExperience"]]
target = train["blueWins"]
test_features = test[["blueFirstBlood", "blueKills", "blueDeaths", "blueAssists", "blueEliteMonsters", "blueDragons", "blueTotalGold", "blueTotalExperience"]]

# 変数の標準化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
test_features_scaled = scaler.transform(test_features)

# モデルの学習
hgb = HistGradientBoostingClassifier(
    loss='log_loss', 
    max_iter=100, 
    max_depth=None, 
    max_leaf_nodes=31, 
    min_samples_leaf=2, 
    learning_rate=0.01, 
    random_state=42
    )
hgb.fit(features_scaled, target)
my_prediction = hgb.predict(test_features_scaled)

# csvの作成
gameId = np.array(test["gameId"]).astype(int)
data = list(zip(gameId, my_prediction))  # 行単位でデータをまとめる
my_solution = pd.DataFrame(data)
my_solution.to_csv("my_tree_one.csv", index=False, header=False)