"""Ground truth generator for lane estimator benchmarking."""
import codecs
import numpy as np
import scipy as sp
from scipy.interpolate import splev, splprep
import yaml


class Lane:
    def __init__(self, filename):
        with codecs.open(filename, "r", encoding="utf-8") as f:
            data = yaml.load(f)
        # Select the lane config only
        cfg = data["lane"]

        # Transpose the arrays (data points as columns)
        self.waypoints_orig = np.array(cfg["waypoints"]).T
        # Get a (cubic) B-spline representation of the points
        self.waypoints_tck, self.u_orig = splprep(
            (self.waypoints_orig[0, :], self.waypoints_orig[1, :]),
            k=3,
            s=cfg["smoothing"],
            )
        # Get the new, finer, spline parameter range
        self.u_interp = np.linspace(0.0, 1.0, cfg["num_interpolated"])
        self.waypoints_interp = np.array(
            splev(self.u_interp, self.waypoints_tck)
            )


class EgoMotion:
    def __init__(self, filename):
        with codecs.open(filename, "r", encoding="utf-8") as f:
            data = yaml.load(f)
        # Select the ego motion config only
        cfg = data["ego_motion"]

        # Transpose the arrays (data points as columns)
        points = np.array(cfg["waypoints"]).T
        self.waypoints_orig = points[:2, :]
        self.yaw_orig = points[2, :]
        # Get a (cubic) B-spline representation of the points
        self.waypoints_tck, self.u_orig = splprep(
            (self.waypoints_orig[0, :], self.waypoints_orig[1, :]),
            k=3,
            s=cfg["smoothing"],
            )
        # Get the new, finer, spline parameter range
        self.u_interp = np.linspace(0.0, 1.0, cfg["num_interpolated"])
        # Interpolate the (x, y) waypoints with a spline
        self.waypoints_interp = np.array(
            splev(self.u_interp, self.waypoints_tck)
            )

        self.yaw_tck = splprep(
            (self.waypoints_orig[2, :],),
            u=self.u_orig,
            s=cfg["smoothing"],
            )
        # Interpolate the yaw points with a spline
        self.yaw_interp = np.array(
            interpolate.splev(self.u_interp, self.yaw_tck)
            )
        self.yaw_interp_der = np.array(
            interpolate.splev(self.u_interp, self.yaw_tck, der=1)
            )


class MeasurementNoise():
    def __init__(self, filename):
        with codecs.open(filename, "r", encoding="utf-8") as f:
            data = yaml.load(f)
        # Select the measurement noise config only
        cfg = data["measurement_noise"]

        self.dof = cfg["radial"]["degrees_of_freedom"]
        self.scaling = cfg["radial"]["scaling"]

    def sample(self, num_samples):
        # Draw the angle from the uniform pdf and
        # the radius from the Student's t-distribution
        rho = np.abs(np.random.standard_t(self.dof, size=num_samples))
        rho *= self.scaling
        phi = np.random.uniform(low=-np.pi, high=np.pi, size=num_samples)
        return rho*np.vstack((np.cos(phi), np.sin(phi)))
