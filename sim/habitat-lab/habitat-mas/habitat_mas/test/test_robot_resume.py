import pytest
import os
import numpy as np
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import IkHelper
from matplotlib import pyplot as plt
from habitat.utils.visualizations.utils import observations_to_image
from habitat.articulated_agents.mobile_manipulator import MobileManipulator
from habitat_mas.agents.capabilities.manipulation import get_arm_workspace
from .data_utils import get_fetch_hssd_env, get_spot_hssd_env


def test_get_arm_workspace():

    # env = get_fetch_hssd_env()
    env = get_spot_hssd_env()
    for i in range(5):
        env.reset()
    sim:RearrangeSim = env.sim
    sim._debug_render = True
    agent:MobileManipulator = sim.agents_mgr[0].articulated_agent
    num_bins = 3
    visualize = True
    # Call the function under test
    center, radius = get_arm_workspace(sim, 0, num_bins, geometry="sphere", visualize=visualize)

    if visualize:
        obs = env._sim.get_sensor_observations()
        obs['third_rgb'] = obs['third_rgb'][:, :, :3]
        obs['head_rgb'] = obs['head_rgb'][:, :, :3]
        info = env.get_metrics()
        render_obs = observations_to_image(obs, info, {}, 0)
        plt.figure()
        plt.imshow(render_obs)

    pass
