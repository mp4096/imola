import numpy as np
from .views import get_car_view, filter_within_frame


def get_measurements(idx, lane, ego_motion, measurement_noise, camera):
    """Generate estimator test data.

    Parameters
    ----------
    idx : int
        index of the current car position (interpolated ego motion waypoints)

    lane : Lane
        lane object

    ego_motion : EgoMotion
        ego motion object

    measurement_noise :  MeasurementNoise
        measurement noise object

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
        lane.waypoints_interp,
        camera.frame_width,
        camera.frame_height,
        ego_motion.waypoints_interp[:, idx],
        [ego_motion.yaw_interp[idx]],
        )

    # Decide how many measurements we have
    num_measurements = measurement_noise.choose_number_of_samples()
    # Get the indices of the detected measurements.
    # Sample from a uniform integer distribution _with replacement_
    measurement_indices = np.random.randint(
        ground_truth.shape[1],
        size=num_measurements,
        )

    # Get the measurements and add noise
    measurements = ground_truth[:, measurement_indices] \
        + measurement_noise.sample_spatial(num_measurements)
    # Filter the measurements so they lie within the camera view frame
    measurements = filter_within_frame(
        measurements,
        camera.frame_width,
        camera.frame_height,
        )

    return measurements, ground_truth
