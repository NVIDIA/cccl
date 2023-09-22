
class Build:
  def __init__(self, code, elapsed):
      self.code = code
      self.elapsed = elapsed

  def __repr__(self):
      return "Build(code = {}, elapsed = {:.4f}s)".format(self.code, self.elapsed)
