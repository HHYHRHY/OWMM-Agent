import numpy as np

class GridMap:
    def __init__(self, time, resolution, width, height, pos, quat, grid):
        self.time = time
        self.resolution = resolution
        self.width = width
        self.height = height
        # the real-world pose of the cell (0,0) in the map.
        self.origin_pos = pos  # (x, y, z)
        self.origin_quat = quat  # quaternion

        self.grid = grid  # numpy array

