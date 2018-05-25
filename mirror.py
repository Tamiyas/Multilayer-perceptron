from mlp import *

class Mirror():
  def __init__(self, column = 6):
    """ミラーシンメトリ問題に使用する入力データと教師データを作成.
    Attributes:
      input: 入力データ.
      train:  教師データ.
    """
    self.input = self.generate_mirror(column)
    self.train = self.check_symmetry()

  def get(self):
    """学習データの作成.
    Returns:
      学習データ(入力データと教師データのタプル).
    """
    return (self.input, self.train)

  def generate_mirror(self, column):
    """2進数列を作成してinputに格納する.
    Args:
      column: ミラーシンメトリを作成する桁数.
    Returns:
      2進数列
    """
    # 2進数をinputに格納(数列の先頭を0パディングする)
    input = [(format(i, 'b').zfill(column)) for i in range(2 ** column)]
    self.str_input = input
    return np.array([list(map(int, X)) for X in input])

  def check_symmetry(self):
    """与えられた文字列が、左右対称かどうかを判定する.
    Returns:
      教師データ.
    """
    center = self.input.shape[1] // 2
    train = np.zeros((self.input.shape[0], 1)).astype('int')
    for (i, X) in enumerate(self.str_input):
      # 文字列の前半部分を取得する
      first = X[:center]
      # 文字列の後半部分を取得する
      last = X[center:]
      train[i] = (first == last)
    return train

if __name__ == '__main__':
  mirror = Mirror()
  mlp = MLP(mirror, hidden = 5)
  mlp.train(epoch = 10000)
  mlp.error_graph()
  mlp.predict(mirror)
