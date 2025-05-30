import rospy 
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Quaternion, Point

# local import 
from .grid_map import GridMap

def from_occmsg_to_gridmap(occ_msg: OccupancyGrid):

    grid = np.asarray(occ_msg.data, dtype=np.int8).reshape(
        occ_msg.info.height, occ_msg.info.width
    )
    grid[grid == 100] = 1
    ros_p = occ_msg.info.origin.position
    ros_q = occ_msg.info.origin.orientation
    return GridMap(
        occ_msg.header.stamp.to_sec(),
        occ_msg.info.resolution,
        occ_msg.info.width,
        occ_msg.info.height,
        np.array([ros_p.x, ros_p.y, ros_p.z]),
        np.quaternion(ros_q.w, ros_q.x, ros_q.y, ros_q.z),
        grid,
    )