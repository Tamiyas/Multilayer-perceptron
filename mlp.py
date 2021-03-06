"""MLP(3層パーセプトロン)の実装.
Author:
  T.Miyaji
Date:
  2018/05/06
References:
  [1] https://qiita.com/ta-ka/items/bcdfd2d9903146c51dcb
  [2] ゼロから作る Deep Learning
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

class MLP:
  """3層パーセプトロン.
  Note:
    学習データdataは、入力データと教師データを返すget()メソッドを作成している必要がある.
  Args:
    data:  学習データ.
    hidden: 中間層の素子数.
    params: 重みパラメータ.
  Attributes:
    X:  入力データ.
    Y:  教師データ.
    params: 重みパラメータ.
  """
  def __init__(self, data, hidden):
    # 入力データXと教師データYを取得.
    (self.X, self.Y) = data.get()
    # 重みパラメータの初期化(Xavierの初期値).
    self.params = {}
    self.params['S'] = self.init_params(self.X.shape[1] + 1, hidden)
    self.params['W'] = self.init_params(hidden + 1, self.Y.shape[1])

  def init_params(self, row, column):
    """重みパラメータの初期化処理の実装.
    Args:
      row:  行数. 入力信号数を指定する.
      column: 列数. 出力信号数を指定する.
    Returns:
      Xavierの初期値.
    """
    return np.random.randn(row, column) / np.sqrt(row)

  def train(self, lr = 0.1, epoch = 100000):
    """学習データの学習を実行する関数. バッチ学習の勾配降下法で学習する.
    Args:
      lr: 学習率.
      epoch:  エポック数. 学習データをすべて使い切ったときの回数.
    Attributes:
      error:  学習回数ごとの損失関数の値を格納するリスト.
    """
    self.error = np.zeros(epoch)
    for t in tqdm(range(epoch)):
      grads = {}
      grads['W'] = np.zeros((self.params['W'].shape[0], self.params['W'].shape[1]))
      grads['S'] = np.zeros((self.params['S'].shape[0], self.params['S'].shape[1]))
      error = 0.0
      for i in range(self.X.shape[0]):
        # i行目の入力データと教師データを取得.
        X = self.X[i, :]
        Y = self.Y[i, :]
        # 順伝播.
        (Z, U) = self.forward(X)
        # 逆伝播.
        self.backward(X, Y, Z, U, grads)
        # 誤差を計算.
        error += self.loss(Z, Y)
      # 重みパラメータの更新.
      self.params['W'] -= lr * grads['W']
      self.params['S'] -= lr * grads['S']
      # 損失関数の値を格納.
      self.error[t] = error

  def forward(self, X):
    """順伝播の実装.
    Args:
      X:  入力データ.
    Returns:
      U:  中間層の出力信号.
      Z:  出力層の出力信号.
    """
    U = self.sigmoid(np.dot(np.r_[np.array([1]), X], self.params['S']))
    Z = self.sigmoid(np.dot(np.r_[np.array([1]), U], self.params['W']))
    return (Z, U)

  def backward(self, X, Y, Z, U, grads):
    """逆伝播の実装.
    Args:
      X:  入力データ.
      Y:  教師データ.
      Z:  出力層の出力信号.
      U:  中間層の出力信号.
      grads: 各層の重みパラメータに関する損失関数の勾配を格納するディクショナリ.
    """
    W = self.params['W']
    # 誤差逆伝播法によって出力層および隠れ層のデルタを求める.
    d_out = (Z - Y) * Z * (1 - Z)
    d_hidden = np.dot(d_out, W[1:, :].T) * U * (1 - U)

    # Wに関する損失関数の勾配を計算.
    grads['W'] += (d_out.reshape((-1, 1)) * np.r_[np.array([1]), U]).T
    # Sに関する損失関数の勾配を計算.
    grads['S'] += (d_hidden.reshape((-1, 1)) * np.r_[np.array([1]), X]).T

    return grads

  def sigmoid(self, x):
    """シグモイド関数の実装.
    Args:
      x:  入力.
    Returns:
      シグモイド関数の出力.
    """
    return 1 / (1 + np.exp(-x))

  def loss(self, Z, Y):
    """2乗和誤差を計算.
    Args:
      Z:  出力層の出力信号.
      Y:  教師データ.
    Returns:
      2乗和誤差.
    """
    return 0.5 * (Z - Y) ** 2

  def predict(self):
    """学習したモデルを使って予測する関数.
    """
    (X, Y) = (self.X, self.Y)
    Z = np.zeros(X.shape[0])
    Correct = 0

    for i in range(X.shape[0]):
      x = X[i, :]
      y = Y[i, :]
      z, _ = self.forward(x)
      Z[i] = z
      c = self.threshold(z)
      if c == y:
        Correct += 1
      print('入力 {0}, 正解 {1}: 出力 {2}'.format(x, y, z))
    print('W:\n', self.params['W'])
    print('S:\n', self.params['S'])
    print('Accuracy {:.2%}'.format(Correct / float(X.shape[0])))

    return Z

  def threshold(self, z):
    """学習が成功しているかどうかをしきい値に従って判定する関数.
    Args:
      z:  出力信号.
    Returns:
      1 if 学習が成功 else 0
    """
    if z >= 0.9:
      return 1
    elif z <= 0.1:
      return 0
    else:
      return z

  def error_graph(self, save_dir = 'figure'):
    """損失関数の推移をグラフで描画する.
    Args:
      save_dir: グラフを保存するディレクトリ名.
    """
    plt.ylim(0.0, self.error.max() + 1.0)
    plt.plot(np.arange(0, len(self.error)), self.error)
    if Path(save_dir).exists() == False:
      Path(save_dir).mkdir()
    plt.savefig(self.file_name(save_dir + '/error_'))
    plt.show()

  def decision_boundary(self, step = 0.02, save_dir = 'figure'):
    """決定境界をプロットする関数.
    Args:
      step: 座標のステップ数.
      save_dir: グラフを保存するディレクトリ名.
    """
    if(self.X.shape[1] != 2):
      return
    (x_min, x_max) = (self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5)
    (y_min, y_max) = (self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5)
    # 格子点の作成
    (xx, yy) = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    X = np.c_[xx.ravel(), yy.ravel()]

    Z = np.zeros(X.shape[0])
    for i in tqdm(range(X.shape[0])):
      z, _ = self.forward(X[i, :])
      Z[i] = np.round(z)

    Z = np.reshape(Z, xx.shape)
    plt.xlim(x_min, x_max)
    # 境界面のプロット
    plt.contourf(xx, yy, Z, cmap = plt.cm.Spectral, alpha = 0.8)
    # 入力データのプロット
    plt.scatter(self.X[:, 0], self.X[:, 1], c = self.Y[:, 0], cmap = plt.cm.Spectral, s = 15)
    plt.colorbar()
    if Path(save_dir).exists() == False:
      Path(save_dir).mkdir()
    plt.savefig(self.file_name(save_dir + '/boundary_'))
    plt.show()

  def file_name(self, prefix):
    """現在時刻を含めたファイル名を返す関数.
    Args:
      prefix: 現在時刻の前に付ける名前.
    Returns:
      ファイル名.
    """
    return prefix + datetime.now().strftime('%Y%m%d-%H%M%S.pdf')
