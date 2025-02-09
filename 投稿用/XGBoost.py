import os
import pandas as pd
import numpy as np
import xgboost as xgb
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
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb_model.fit(features_scaled, target)
my_prediction = xgb_model.predict(test_features_scaled)

# csvの作成
gameId = np.array(test["gameId"]).astype(int)
data = list(zip(gameId, my_prediction))  # 行単位でデータをまとめる
my_solution = pd.DataFrame(data)
my_solution.to_csv("my_tree_one.csv", index=False, header=False)