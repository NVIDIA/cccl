class Build:
    def __init__(self, code, elapsed):
        self.code = code
        self.elapsed = elapsed

    def __repr__(self):
        return f"Build(code = {self.code}, elapsed = {self.elapsed:.4f}s)"
