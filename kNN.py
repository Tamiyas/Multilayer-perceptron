"""k-NN(k近傍法, k-Nearest Neighbors)の実装.
Author:
  T.Miyaji
Date:
  2018/05/28
References:
  http://blog.amedama.jp/entry/2017/03/18/140238
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class kNN:
  """k-NN.
  Args:
    data: 学習データ.
    k:  分類に用いる近傍の数.
  """
  def __init__(self, data, k = 1):
    (self.X, self.Y) = data.get()
    self.k = k

  def predict(self):
    """k-NNアルゴリズムを用いてクラス(0 or 1)を予測する関数.
    """
    Correct = 0
    for (i, x) in enumerate(self.X):
      voted_class = self.nearest_neighbor(x)
      if voted_class == self.Y[i, :]:
        Correct += 1
      print('入力 {0}, 正解 {1}, 出力{2}'.format(x, self.Y[i, :], voted_class))
      print('Accuracy {:.2%}'.format(Correct / float(self.X.shape[0])))

  def nearest_neighbor(self, x):
    """k-NNアルゴリズムの実装.
    Args:
      x:  注目点.
    Returns:
      近傍の点でクラスの多数決をして最も多いクラス(0, 1)を返す.
    """
    # 教師データの点pと注目点xの距離のベクトルを作成する.
    distances = np.array([self.distance(p, x) for p in self.X])
    # 距離が近い順にソートしてk個インデックスを得る
    nearest_indexes = distances.argsort()[:self.k]
    # 取得したk個のインデックスのクラスを得る
    nearest_classes = self.Y[nearest_indexes]
    # 取得したk個の中で最も多いクラスを返す
    return self.majority_vote(nearest_classes)

  def distance(self, x1, x2):
    """2点間の距離を計算する関数.
    Note:
      今回の実装はユークリッド距離だが, マンハッタン距離でも問題ない.
      さらに、ユークリッド距離の2乗でも問題ない
    Args:
      x1: 2次元空間の座標.
      x2: 2次元空間の座標.
    Returns:
      ユークリッド距離.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

  def majority_vote(self, classes):
    """リストの中で最も出現する値を返す関数.
    Args:
      classes:  クラス(0 or 1)が格納されたリスト.
    Returns:
      クラス(0 or 1).
    """
    return 0 if (np.sum(classes == 0) > np.sum(classes == 1)) else 1

  def decision_boundary(self, step = 0.02):
    """決定境界をプロットする関数.
    Args:
      step: 座標のステップ数.
    """
    if(self.X.shape[1] != 2):
      return

    (x_min, x_max) = (self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5)
    (y_min, y_max) = (self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5)
    # 格子点の作成
    (xx, yy) = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    X = np.c_[xx.ravel(), yy.ravel()]

    Z = np.array([self.nearest_neighbor(X[i, :]) for i in tqdm(range(X.shape[0]))])
    Z = np.reshape(Z, xx.shape)
    plt.xlim(x_min, x_max)
    # 境界面のプロット
    plt.contourf(xx, yy, Z, cmap = plt.cm.Spectral, alpha = 0.8)
    # 入力データのプロット
    plt.scatter(self.X[:, 0], self.X[:, 1], c = self.Y[:, 0], cmap = plt.cm.Spectral, s = 15)
    plt.colorbar()
    plt.show()
