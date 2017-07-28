"""Ground truth generator for lane estimator benchmarking."""
import codecs
import numpy as np
from scipy.interpolate import splev, splprep
import yaml
from .views import filter_within_frame


def load_yaml(filename):
    with codecs.open(filename, "r", encoding="utf-8") as f:
        data = yaml.load(f)
    return data


class Lane:
    def __init__(self, data):
        cfg = data["lane"]
        # Transpose the array with (x, y) points as columns
        self.xy_coarse = np.array(cfg["waypoints"]).T
        # Get a (cubic) B-spline representation of the points
        self.xy_tck, self.u_coarse = splprep(
            (self.xy_coarse[0, :], self.xy_coarse[1, :]),
            k=3,
            s=cfg["smoothing"],
            )
        # Get the new, finer, spline parameter range
        self.u = np.linspace(0.0, 1.0, cfg["num_interpolated"])
        self.xy = np.array(splev(self.u, self.xy_tck))


class EgoMotion:
    def __init__(self, data):
        cfg = data["ego_motion"]
        # Transpose the waypoints array (data points as columns)
        points = np.array(cfg["waypoints"]).T

        # Process the time points
        self.t_coarse = points[0, :]
        if not np.all(self.t_coarse[:-1] < self.t_coarse[1:]):
            raise ValueError("Timestamps must be strictly increasing.")

        # Process the (x, y) points
        self.xy_coarse = points[1:3, :]
        # Get a (cubic) B-spline representation of the points
        self.xy_tck, _ = splprep(
            (self.xy_coarse[0, :], self.xy_coarse[1, :]),
            u=self.t_coarse,
            k=3,
            s=cfg["smoothing"],
            )
        # Get the new, finer, spline parameter range
        self.t = np.linspace(
            self.t_coarse[0],
            self.t_coarse[-1],
            cfg["num_interpolated"],
            )
        # Interpolate the (x, y) points with a spline
        self.xy = np.array(splev(self.t, self.xy_tck))

        # Get the ego motion yaw
        self.xy_der = np.array(splev(self.t, self.xy_tck, der=1))
        normalised_der = self.xy_der/np.linalg.norm(self.xy_der, axis=0)
        self.yaw = np.arctan2(normalised_der[1, :], normalised_der[0, :])

        # Compute the ego motion yaw derivative
        #
        # Rationale: If we compute the yaw derivative using the naive approach
        # (by differentiating yaw angle), we shall run into problems when
        # the yaw angles jumps between -π and π.
        #
        # \dot{\phi}
        # =
        # \frac{-\dot{y} \ddot{x} + \dot{x} \ddot{y}}{\dot{x}^2 + \dot{y}^2}
        #
        xy_der2 = np.array(splev(self.t, self.xy_tck, der=2))
        self.xy_der2 = xy_der2
        xy_der_squared_norm = (self.xy_der**2).sum(axis=0)
        self.yaw_der = \
            -self.xy_der[1, :]*xy_der2[0, :] + self.xy_der[0, :]*xy_der2[1, :]
        self.yaw_der /= xy_der_squared_norm

        # Process ego motion yaw deviation
        self.yaw_deviation_coarse = points[3, :]
        self.yaw_deviation_tck, _ = splprep(
            (self.yaw_deviation_coarse,),
            u=self.t_coarse,
            s=cfg["smoothing_yaw_deviation"],
            )
        # Interpolate the yaw deviation points with a spline
        self.yaw_deviation = np.squeeze(
            splev(self.t, self.yaw_deviation_tck),
            )
        self.yaw_deviation_der = np.squeeze(
            splev(self.t, self.yaw_deviation_tck, der=1),
            )

    def get_velocity(self, idx):
        return self.xy_der[:, idx]

    def get_angular_velocity(self, idx):
        return self.yaw_der[idx] + self.yaw_deviation_der[idx]

    def find_index(self, time):
        return np.searchsorted(self.t, time)


class MeasurementNoiseCamera():
    def __init__(self, data):
        cfg = data["measurement_noise_camera"]
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


class MeasurementNoiseImu():
    def __init__(self, data):
        cfg = data["measurement_noise_imu"]
        self.velocity_std = np.sqrt(cfg["velocity_variance"])
        self.angular_velocity_std = np.sqrt(cfg["angular_velocity_variance"])

    def sample_velocity(self):
        return np.random.randn(2)*self.velocity_std

    def sample_angular_velocity(self):
        return np.random.randn(1)*self.angular_velocity_std


class Camera():
    def __init__(self, data):
        cfg = data["camera"]
        self.frame_width = cfg["frame_width"]
        self.frame_height = cfg["frame_height"]

    def visible_points(self, points):
        return filter_within_frame(points, self.frame_width, self.frame_height)
