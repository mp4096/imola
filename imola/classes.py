"""Ground truth generator for lane estimator benchmarking."""
import codecs
import numpy as np
from scipy.interpolate import splev, splprep
import yaml


def load_yaml(filename):
    with codecs.open(filename, "r", encoding="utf-8") as f:
        data = yaml.load(f)
    return data


class Lane:
    def __init__(self, data):
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
    def __init__(self, data):
        cfg = data["ego_motion"]
        # Transpose the arrays (data points as columns)
        points = np.array(cfg["waypoints"]).T
        self.waypoints_orig = points[:2, :]
        self.yaw_deviation_orig = points[2, :]
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

        self.yaw_deviation_tck, _ = splprep(
            (self.yaw_deviation_orig,),
            u=self.u_orig,
            s=cfg["smoothing_yaw_deviation"],
            )
        # Interpolate the yaw points with a spline
        self.yaw_deviation_interp = np.squeeze(
            splev(self.u_interp, self.yaw_deviation_tck),
            )
        self.yaw_deviation_interp_der = np.squeeze(
            splev(self.u_interp, self.yaw_deviation_tck, der=1),
            )

        # Store index velocity
        self.velocity = cfg["velocity"]


class MeasurementNoise():
    def __init__(self, data):
        cfg = data["measurement_noise"]
        self.dof = cfg["radial"]["degrees_of_freedom"]
        self.scaling = cfg["radial"]["scaling"]
        self.expected_num = cfg["expected_number_of_measurements"]

    def choose_number_of_samples(self):
        # Decide how many detected measurements we should have.
        # Draw this number from the Poisson distribution,
        # it seems to be a good fit for this kind of modelling;
        # however, one assumption seems to be violated
        # (independence of subsequent trials).
        # See:
        # * https://en.wikipedia.org/wiki/Poisson_distribution
        # * https://en.wikipedia.org/wiki/Negative_binomial_distribution
        return np.random.poisson(self.expected_num)

    def sample_spatial(self, num_samples):
        # Draw the angle from the uniform pdf and
        # the radius from the Student's t-distribution
        rho = np.abs(np.random.standard_t(self.dof, size=num_samples))
        rho *= self.scaling
        phi = np.random.uniform(low=-np.pi, high=np.pi, size=num_samples)
        return rho*np.vstack((np.cos(phi), np.sin(phi)))


class Camera():
    def __init__(self, data):
        cfg = data["camera"]
        self.frame_width = cfg["frame_width"]
        self.frame_height = cfg["frame_height"]
