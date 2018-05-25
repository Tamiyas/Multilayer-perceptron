from mlp import *
import numpy as np
import matplotlib.pyplot as plt

class Sinusoid():
  def __init__(self, xlim = (-6, 6), ylim = (-1.5, 1.5), num = 100, noise = True):
    """ノイズが乗った正弦関数の入力データと教師データを作成.
    Attributes:
      input:  入力データ.
      train:  教師データ.
    """
    domain = np.array(np.linspace(xlim[0], xlim[1], num))
    if(noise == True):
      self.input = self.generate_noise(domain, ylim, num)
    else:
      self.input = self.generate_no_noise(domain, ylim, num)

    self.train = self.check_input()

  def get(self):
    """学習データの作成.
    Returns:
      学習データ(入力データと教師データのタプル).
    """
    return (self.input, self.train)

  def generate_noise(self, domain, ylim, num):
    """ノイズデータを作成する.
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

  def generate_no_noise(self, domain, ylim, num):
    """ノイズなしデータを作成する.
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

if __name__ == '__main__':
  sinusoid = Sinusoid()
  mlp = MLP(sinusoid, hidden = 15)
  mlp.train(epoch = 15000)
  mlp.predict()
  mlp.error_graph()
  domain = np.array(np.linspace(-6.5, 6.5))
  plt.plot(domain, np.sin(np.pi * domain / 2), lw = 1.5, color = 'black')
  mlp.decision_boundary()
