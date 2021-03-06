from mlp import *

class XOR:
  """XOR学習データ.
  Attributes:
    input: 入力データ.
    train:  教師データ.
  """
  def __init__(self):
    # XOR問題に使用する入力データと教師データを作成
    self.input = np.array([[0, 0],
                          [0, 1],
                          [1, 0],
                          [1, 1]])
    self.train = np.array([[0], [1], [1], [0]])

  def get(self):
    """学習データの作成
    Returns:
      学習データ(入力データと教師データのタプル).
    """
    return (self.input, self.train)

if __name__ == '__main__':
  xor = XOR()
  mlp = MLP(xor, hidden = 2)
  mlp.train(epoch = 15000)
  mlp.predict()
  mlp.error_graph()
  mlp.decision_boundary()
