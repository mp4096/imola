"""Ground truth generator for lane estimator benchmarking."""

import codecs
import numpy as np
import scipy as sp
from scipy.interpolate import splev, splprep, splrep
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


def interpolate_points(points, num_interpolated, smoothing=0.05):
    """Interpolate planar points with a spline.

    Given ``num_original`` points specified by their (x, y)-coordinates,
    interpolate between them using a cubic spline and
    return ``num_interpolated`` new points.

    Parameters
    ----------
    points : (2, num_original) array_like
        original points (as column vectors)

    num_interpolated : int
        number of the interpolated points. Should be a positive integer
        and greater than the number of original points

    smoothing : float, optional
        smoothing condition for the spline interpolation,
        see documentation for ``scipy.interpolate.splprep``

    Returns
    -------
    (2, num_interpolated) ndarray
        interpolated points

    """
    points = np.array(points)
    # Get a (cubic) B-spline representation of the (x, y)-points
    tck, _ = splprep((points[0, :], points[1, :]), k=3, s=smoothing)
    # Get the new, finer, spline parameter range
    parameter_range = np.linspace(0.0, 1.0, num_interpolated)
    return np.array(splev(parameter_range, tck))


def interpolate_points_derivative(points, num_interpolated, smoothing=0.05):
    """Compute derivative (tangential) along a spline defined by planar points.

    Given ``num_original`` points specified by their (x, y)-coordinates,
    interpolate between them using a cubic spline and
    return the tangential vector at ``num_interpolated`` new points.

    Parameters
    ----------
    points : (2, num_original) array_like
        original points (as column vectors)

    num_interpolated : int
        number of the interpolated points. Should be a positive integer
        and greater than the number of original points

    smoothing : float, optional
        smoothing condition for the spline interpolation,
        see documentation for ``scipy.interpolate.splprep``

    Returns
    -------
    (2, num_interpolated) ndarray
        derivative of the interpolated spline

    """
    points = np.array(points)
    # Get a (cubic) B-spline representation of the (x, y)-points
    tck, _ = splprep((points[0, :], points[1, :]), k=3, s=smoothing)
    # Get the new, finer, spline parameter range
    parameter_range = np.linspace(0.0, 1.0, num_interpolated)
    return np.array(splev(parameter_range, tck, der=1))


def interpolate_points_yaw(points, num_interpolated, smoothing=0.05):
    """Compute the yaw along the spline defined by planar points.

    Given ``num_original`` points specified by their (x, y)-coordinates,
    interpolate between them using a cubic spline and
    at ``num_interpolated`` new points. Now compute the angle of the
    tangential vecotr at each interpolated point.

    Parameters
    ----------
    points : (2, num_original) array_like
        original points (as column vectors)

    num_interpolated : int
        number of the interpolated points. Should be a positive integer
        and greater than the number of original points

    smoothing : float, optional
        smoothing condition for the spline interpolation,
        see documentation for ``scipy.interpolate.splprep``

    Returns
    -------
    (num_interpolated,) ndarray
        yaw at interpolated points

    """
    der = interpolate_points_derivative(
        points,
        num_interpolated,
        smoothing=smoothing,
        )
    der_normalised = der/np.linalg.norm(der, axis=0)
    return np.arctan2(der_normalised[1, :], der_normalised[0, :])

