import time

import numpy as np


class Fps:
    def __init__(self, estimaeted_fps=30.0, nr_of_avg_frames=50):
        self.ctr = 0
        self.time = time.time()
        self.time_old = time.time()
        self.nr_of_avg_frames = nr_of_avg_frames
        self.estimaeted_fps = estimaeted_fps
        self.time_array = None
        self.reset()
        self.fps = 1 / (np.mean(self.time_array) + 0.00001)

    def reset(self):
        self.time_array = np.ones(self.nr_of_avg_frames) * 1 / self.estimaeted_fps

    def update(self):
        self.time_array[int(self.ctr)] = time.time() - self.time_old
        self.time_old = time.time()
        self.ctr += 1
        if self.ctr > (self.nr_of_avg_frames - 1):
            self.ctr = 0
        self.fps = 1 / (np.mean(self.time_array) + 0.00001)

    def delay(self):
        time.sleep(1/self.fps)