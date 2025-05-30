import os


class SceneGraphHabitatConfig:
    def __init__(self) -> None:

        # dataset config
        self.dataset = "habitat"
        self.model = "3DSSG"
        self.load_ground_truth = True  # load gt from dataset
        self.meters_per_grid = 0.1  # free space grid resolution
        self.object_grid_scale = 1
        self.floor_heights_file = "metadata/floor_heights.json"
        assert os.path.exists(self.floor_heights_file)
        # self.height = 0 # if not set, dynamically get height from scene bbox

        # object node config
        self.free_space = True
        self.aligned_bbox = True  # use aabb if true, otherwise use obb
        self.point_clouds = True  # segment point clouds for all object nodes
        self.object_gcn_feature = True

        # relationship config
        self.rel_gcn_feature = True
        self.rel_geometric = True
