from mlp import *
from kNN import *
import numpy as np
import matplotlib.pyplot as plt

class Sinusoid():
  def __init__(self, xlim = (-6, 6), ylim = (-1.5, 1.5), num = 100, random = True):
    """ノイズが乗った正弦関数の入力データと教師データを作成.
    Attributes:
      input:  入力データ.
      train:  教師データ.
    """
    domain = np.array(np.linspace(xlim[0], xlim[1], num))
    if(random == True):
      self.input = self.generate_random(domain, ylim, num)
    else:
      self.input = self.generate_no_random(domain, ylim, num)

    self.train = self.check_input()

  def get(self):
    """学習データの作成.
    Returns:
      学習データ(入力データと教師データのタプル).
    """
    return (self.input, self.train)

  def generate_random(self, domain, ylim, num):
    """ノイズがランダムなデータを作成する.
    Args:
      domain: 定義域.
      ylim: 値域.
      num:  入力データ数.
    Returns:
      ノイズデータ.
    """
    # ylimの範囲で一様乱数をnum個生成する.
    image = np.array((ylim[1] - ylim[0]) * np.random.rand(num) + ylim[0])

    return np.c_[domain, image]

  def generate_no_random(self, domain, ylim, num):
    """ノイズがランダムでないデータを作成する.
    Args:
      domain: 定義域.
      ylim: 値域.
      num:  入力データ数.
    Returns:
      ノイズなしデータ.
    """
    x = np.random.choice(domain, num, replace = False)
    above = np.array(np.sin(np.pi * x[:num // 2] / 2) + 0.9)
    below = np.array(np.sin(np.pi * x[num // 2:] / 2) - 0.9)
    image = np.r_[above, below]

    return np.c_[x, image]

  def check_input(self):
    """入力データが正弦関数sin(pi / 2)の上にあるかどうかを判定する.
    Returns:
      教師データ.
    """
    domain = self.input[:,0]
    # 正弦関数sin(pi / 2)を作成する.
    sinusoid = np.sin(np.pi * domain / 2)
    image = np.array(self.input[:,1] > sinusoid, dtype=int)
    return np.reshape(image, (domain.shape[0], 1))

  def plot_artificial_data(self, xlim = (-6, 6), ylim = (-1.5, 1.5), num = 100):
    """作成した入力データをグラフでプロットする.
    """
    plt.xlim(xlim)
    plt.ylim(ylim)
    for point in self.input:
      plt.scatter(point[0], point[1], c = 'lightgreen')
    domain = np.array(np.linspace(xlim[0], xlim[1], num))
    plt.plot(domain, np.sin(np.pi * domain / 2))
    plt.show()

  def bias_variance(self, Z):
    """バイアスとバリアンスを計算する.
    Args:
      Z:  学習モデルの出力信号のベクトル.
    """
    (X, Y) = (self.input, self.train)
    # バイアス(学習データと出力信号との差)を計算
    bias = np.average(X[:, 1] - Z) ** 2
    # 分散(出力信号のバラつき)を計算
    variance = np.var(Z)

    print('Bias:{}'.format(bias))
    print('Variance:{}'.format(variance))



if __name__ == '__main__':
  # sinusoid = Sinusoid()
  # # sinusoid = Sinusoid(random = False)
  # mlp = MLP(sinusoid, hidden = 15)
  # mlp.train(epoch = 15000)
  # Z = mlp.predict()
  # sinusoid.bias_variance(Z)
  # mlp.error_graph()
  # domain = np.array(np.linspace(-6.5, 6.5))
  # plt.plot(domain, np.sin(np.pi * domain / 2), lw = 1.5, color = 'black')
  # mlp.decision_boundary()
  sinusoid = Sinusoid()
  knn = kNN(sinusoid, k = 2)
  knn.predict()
  domain = np.array(np.linspace(-6.5, 6.5))
  plt.plot(domain, np.sin(np.pi * domain / 2), lw = 1.5, color = 'black')
  knn.decision_boundary()
