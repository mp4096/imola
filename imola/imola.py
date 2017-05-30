import codecs
import numpy as np
import yaml


def load_scenario(filename):
    r"""Load a scenario (lane waypoints and ego motion data) from file.

    Parameters
    ----------
    filename : str
        path to the YAML file from which the scenario will be loaded

    Returns
    -------
    lane : (3, num_lane_waypoints) ndarray
        lane waypoints, specified as :math:`(x, y)` points
        (each point is a column vector)

    ego_motion : (3, num_ego_motion_points) ndarray
        ego motion data, specified as :math:`(x, y, \phi)` points
        (each point is a column vector)

    """
    with codecs.open(filename, "r", encoding="utf-8") as f:
        data = yaml.load(f)

    # Transpose the arrays (data points as columns)
    lane = np.array(data["lane"]).T
    ego_motion = np.array(data["ego_motion"]).T

    return lane, ego_motion

