from collections import deque, Counter

class LabelSmoother:
    def __init__(self, buffer_size=15):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, new_label):
        self.buffer.append(new_label)
        most_common = Counter(self.buffer).most_common(1)
        return most_common[0][0] if most_common else new_label