import os
import pandas as pd
import numpy as np
import lightgbm as lgb
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
lgb_model = lgb_model = lgb.LGBMClassifier(num_iterations=1000, max_depth=6, random_state=42)
lgb_model = lgb_model.fit(features_scaled, target)
my_prediction = lgb_model.predict(test_features_scaled)

# csvの作成
gameId = np.array(test["gameId"]).astype(int)
data = list(zip(gameId, my_prediction))  # 行単位でデータをまとめる
my_solution = pd.DataFrame(data)
my_solution.to_csv("my_tree_one.csv", index=False, header=False)