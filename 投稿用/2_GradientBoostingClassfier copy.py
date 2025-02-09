import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
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
gd = GradientBoostingClassifier(
    loss='log_loss', 
    learning_rate=0.2, 
    n_estimators=100, 
    subsample=1.0, 
    criterion='friedman_mse', 
    max_depth=3, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    max_features='sqrt', 
    max_leaf_nodes=None, 
    random_state=42)  # ハイパーパラメータを調整
gd.fit(features_scaled, target)
my_prediction = gd.predict(test_features_scaled)

# csvの作成
gameId = np.array(test["gameId"]).astype(int)
data = list(zip(gameId, my_prediction))  # 行単位でデータをまとめる
my_solution = pd.DataFrame(data)
my_solution.to_csv("my_tree_one.csv", index=False, header=False)