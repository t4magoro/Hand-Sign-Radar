# viewer.py
import matplotlib.pyplot as plt
import numpy as np

class HeatmapViewer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(np.zeros((64, 64, 3)), vmin=0, vmax=1)
        self.ax.axis('off')
        self.fig.canvas.manager.set_window_title("Real-Time RGB Heatmap")
        plt.ion()  # Enable interactive mode
        plt.show()

    def update(self, heatmap_rgb):
        self.im.set_data(heatmap_rgb)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
