from collections import deque, Counter

# Label smoothing class to reduce flickering in predictions
class LabelSmoother:
    def __init__(self, buffer_size=15):
        self.buffer = deque(maxlen=buffer_size)  # Fixed-size buffer

    # Update buffer with new label and return the smoothed label
    def update(self, new_label):
        self.buffer.append(new_label)
        most_common = Counter(self.buffer).most_common(1)   # Get the most common label in the buffer
        return most_common[0][0] if most_common else new_label 