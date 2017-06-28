from .transformations import inertial_to_body_frame, body_to_inertial_frame
import numpy as np


def get_view_frame_inertial(frame_width, frame_height,
                            translation_vec, rotations):
    """
    `vf` draws the view frame in the body fixed frame:
                y_body ^
                       │
                       │
              vf[:, 4] ╔═════════════════╗ vf[:, 3] ───────────
                       ║     vf[:, 6]    ║                   ^
    vf[:, 0], vf[:, 5] ╠════════o        ║ ──────> x_body    │ frame_width
                       ║                 ║                   v
              vf[:, 1] ╚═════════════════╝ vf[:, 2] ───────────
                       │                 │
                       │<───────────────>│
                          frame_height
    """

    vf = np.empty((2, 7), dtype=np.float64)
    vf[:, 0] = np.array([0.0, 0.0])
    vf[:, 1] = np.array([0.0, -0.5*frame_width])
    vf[:, 2] = np.array([frame_height, -0.5*frame_width])
    vf[:, 3] = np.array([frame_height, 0.5*frame_width])
    vf[:, 4] = np.array([0.0, 0.5*frame_width])
    vf[:, 5] = np.array([0.0, 0.0])
    vf[:, 6] = np.array([0.5*frame_height, 0.0])
    return inertial_to_body_frame(vf, translation_vec, rotations)


def get_car_view(lane_inertial_frame, frame_width, frame_height,
                 translation_vec, rotations):
    lane_body_frame = body_to_inertial_frame(
        lane_inertial_frame,
        translation_vec,
        rotations,
        )
    return filter_within_frame(lane_body_frame, frame_width, frame_height)


def filter_within_frame(points, frame_width, frame_height):
    """Filter points that lie within a frame.

    Frame specification:
    y_body ^
           │
                   left
           ╔═════════════════╗────────────────────
           ║                 ║                  ^
     lower ║                 ║ upper  ─> x_body │ frame_width
           ║                 ║                  v
           ╚═════════════════╝────────────────────
           │      right      │
           │                 │
           │<───────────────>│
              frame_height
    """
    lower = 0.0 <= points[0, :]
    upper = points[0, :] <= frame_height
    left = points[1, :] <= 0.5*frame_width
    right = -0.5*frame_width <= points[1, :]
    return points[:, left & right & lower & upper]
