from collections import defaultdict


class AvgMeter:
    def __init__(self):
        self.value = 0.
        self.n = 0

    def add(self, value):
        self.value += value
        self.n += 1

    def compute(self):
        return self.value / self.n


class AvgDictMeter:
    def __init__(self):
        self.values = defaultdict(float)
        self.n = 0

    def add(self, values: dict):
        for key, value in values.items():
            self.values[key] += value
        self.n += 1

    def compute(self):
        return {key: value / self.n for key, value in self.values.items()}
