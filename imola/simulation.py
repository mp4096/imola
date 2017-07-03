import numpy as np
from .views import get_car_view, filter_within_frame


def get_measurements_camera(idx, lane, ego_motion, noise_camera, camera):
    """Generate test data from the camera.

    Parameters
    ----------
    idx : int
        index of the current car position (interpolated ego motion waypoints)

    lane : Lane
        lane object

    ego_motion : EgoMotion
        ego motion object

    noise_camera :  MeasurementNoiseCamera
        camera / image processing measurement noise object

    camera : Camera
        camera object

    Returns
    -------
    ground_truth : ndarray
        true points of the lane curve at the given car position
        TODO: How to compute error if we get multiple curve segments
        within one image?

    measurements : ndarray
        measurements at the given car position

    """
    ground_truth = get_car_view(
        lane.xy,
        camera.frame_width,
        camera.frame_height,
        ego_motion.xy[:, idx],
        [ego_motion.yaw[idx], ego_motion.yaw_deviation[idx]],
        )

    # Decide how many measurements we have
    num_measurements = noise_camera.choose_number_of_samples()
    # Get the indices of the detected measurements.
    # Sample from a uniform integer distribution _with replacement_
    measurement_indices = np.random.randint(
        ground_truth.shape[1],
        size=num_measurements,
        )

    # Get the measurements and add noise
    measurements = ground_truth[:, measurement_indices] \
        + noise_camera.sample_spatial(num_measurements)
    # Filter the measurements so they lie within the camera view frame
    measurements = filter_within_frame(
        measurements,
        camera.frame_width,
        camera.frame_height,
        )

    return measurements, ground_truth


def get_measurements_imu(idx, ego_motion, noise_imu):
    """Generate test data from the IMU.

    Parameters
    ----------
    idx : int
        index of the current car position (interpolated ego motion waypoints)

    ego_motion : EgoMotion
        ego motion object

    noise_imu :  MeasurementNoiseImu
        camera / image processing measurement noise object

    Returns
    -------
    ground_truth_velocity : (2,) ndarray
        true velocity

    ground_truth_angular_velocity : float
        true angular velocity

    measurements_velocity : (2,) ndarray
        noisy velocity measurements

    measurements_angular_velocity : float
        noisy angular velocity measurements

    """
    ground_truth_velocity = ego_motion.get_velocity(idx)
    ground_truth_angular_velocity = ego_motion.get_angular_velocity(idx)
    measurements_velocity = \
        ground_truth_velocity + noise_imu.sample_velocity()
    measurements_angular_velocity = \
        ground_truth_angular_velocity + noise_imu.sample_angular_velocity()

    return (
        ground_truth_velocity,
        ground_truth_angular_velocity,
        measurements_velocity,
        measurements_angular_velocity,
        )
