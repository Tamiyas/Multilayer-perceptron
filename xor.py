from mlp import *

class XOR:
  """XOR学習データ.
  Attributes:
    data: 入力データ.
    train:  教師データ.
  """
  def __init__(self):
    # XOR問題に使用する入力データと教師データを作成
    self.data = np.array([[0, 0],
                          [0, 1],
                          [1, 0],
                          [1, 1]])
    self.train = np.array([[0], [1], [1], [0]])

  def get(self):
    """学習データの作成
    Returns:
      学習データ(入力データと教師データのタプル).
    """
    return (self.data, self.train)

if __name__ == '__main__':
  xor = XOR()
  mlp = MLP(xor, hidden = 2)
  mlp.train(epoch = 100000)
  mlp.error_graph()
  mlp.predict(xor)
