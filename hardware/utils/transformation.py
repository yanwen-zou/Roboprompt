"""
Transformation utilities.
"""

import numpy as np

VALID_ROTATION_REPRESENTATIONS = [
    "axis_angle",
    "euler_angles",
    "quaternion",
    "matrix",
    "rotation_6d",
]
ROTATION_REPRESENTATION_DIMS = {
    "axis_angle": 3,
    "euler_angles": 3,
    "quaternion": 4,
    "matrix": 9,
    "rotation_6d": 6,
}


def _normalize(vec, axis=-1, eps=1e-8):
    norm = np.linalg.norm(vec, axis=axis, keepdims=True)
    if np.any(norm < eps):
        raise ValueError("Cannot normalize a zero-length vector.")
    return vec / norm


def _skew_symmetric(vec):
    x, y, z = np.moveaxis(vec, -1, 0)
    zeros = np.zeros_like(x)
    return np.stack(
        [
            np.stack([zeros, -z, y], axis=-1),
            np.stack([z, zeros, -x], axis=-1),
            np.stack([-y, x, zeros], axis=-1),
        ],
        axis=-2,
    )


def _axis_angle_to_matrix(axis_angle):
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    small = theta < 1e-8
    axis = np.divide(axis_angle, theta, out=np.zeros_like(axis_angle), where=~small)

    eye = np.broadcast_to(np.eye(3, dtype=np.float32), axis.shape[:-1] + (3, 3))
    k = _skew_symmetric(axis)
    sin_theta = np.sin(theta)[..., None]
    cos_theta = np.cos(theta)[..., None]
    outer = axis[..., :, None] * axis[..., None, :]
    rot = cos_theta * eye + (1.0 - cos_theta) * outer + sin_theta * k
    if np.any(small):
        rot = np.where(small[..., None], eye, rot)
    return rot.astype(np.float32)


def _matrix_to_axis_angle(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    trace = np.trace(matrix, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    rx = matrix[..., 2, 1] - matrix[..., 1, 2]
    ry = matrix[..., 0, 2] - matrix[..., 2, 0]
    rz = matrix[..., 1, 0] - matrix[..., 0, 1]
    axis_raw = np.stack([rx, ry, rz], axis=-1)
    sin_theta = np.sin(theta)

    scale = np.divide(
        theta,
        2.0 * sin_theta,
        out=np.zeros_like(theta),
        where=np.abs(sin_theta) > 1e-8,
    )
    axis_angle = axis_raw * scale[..., None]
    small = np.abs(theta) < 1e-8
    if np.any(small):
        axis_angle = np.where(small[..., None], np.zeros_like(axis_angle), axis_angle)
    return axis_angle.astype(np.float32)


def _quaternion_to_matrix(quaternion):
    quaternion = _normalize(np.asarray(quaternion, dtype=np.float32))
    w, x, y, z = np.moveaxis(quaternion, -1, 0)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.stack(
        [
            np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1),
            np.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=-1),
            np.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32)


def _matrix_to_quaternion(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices with shape (..., 3, 3), got {matrix.shape}.")

    flat = matrix.reshape(-1, 3, 3)
    quats = []
    for m in flat:
        trace = float(np.trace(m))
        if trace > 0.0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        quats.append([w, x, y, z])

    quats = np.asarray(quats, dtype=np.float32).reshape(matrix.shape[:-2] + (4,))
    return _normalize(quats)


def _euler_angles_to_matrix(euler_angles, convention):
    if convention != "XYZ":
        raise NotImplementedError(f"Euler convention {convention!r} is not implemented.")
    euler_angles = np.asarray(euler_angles, dtype=np.float32)
    x, y, z = np.moveaxis(euler_angles, -1, 0)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)

    return np.stack(
        [
            np.stack([cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy], axis=-1),
            np.stack([cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx], axis=-1),
            np.stack([-sy, cy * sx, cx * cy], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32)


def _matrix_to_euler_angles(matrix, convention):
    if convention != "XYZ":
        raise NotImplementedError(f"Euler convention {convention!r} is not implemented.")
    matrix = np.asarray(matrix, dtype=np.float32)

    sy = -matrix[..., 2, 0]
    y = np.arcsin(np.clip(sy, -1.0, 1.0))
    cy = np.cos(y)
    singular = np.abs(cy) < 1e-6

    x = np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2])
    z = np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])

    x_singular = np.arctan2(-matrix[..., 1, 2], matrix[..., 1, 1])
    z_singular = np.zeros_like(z)

    x = np.where(singular, x_singular, x)
    z = np.where(singular, z_singular, z)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _rotation_6d_to_matrix(rotation_6d):
    rotation_6d = np.asarray(rotation_6d, dtype=np.float32)
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected rotation_6d with shape (..., 6), got {rotation_6d.shape}.")
    a1 = rotation_6d[..., 0:3]
    a2 = rotation_6d[..., 3:6]
    b1 = _normalize(a1)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize(a2 - proj)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def _matrix_to_rotation_6d(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,)).astype(np.float32)


def _to_matrix(rot, from_rep, from_convention=None):
    if from_rep == "matrix":
        matrix = np.asarray(rot, dtype=np.float32)
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(f"Expected matrix rotation with shape (..., 3, 3), got {matrix.shape}.")
        return matrix
    if from_rep == "quaternion":
        return _quaternion_to_matrix(rot)
    if from_rep == "rotation_6d":
        return _rotation_6d_to_matrix(rot)
    if from_rep == "axis_angle":
        return _axis_angle_to_matrix(rot)
    if from_rep == "euler_angles":
        if from_convention is None:
            raise ValueError("Euler-angle input requires from_convention.")
        return _euler_angles_to_matrix(rot, from_convention)
    raise NotImplementedError(f"Rotation representation {from_rep!r} is not implemented.")


def _from_matrix(matrix, to_rep, to_convention=None):
    if to_rep == "matrix":
        return matrix.astype(np.float32)
    if to_rep == "quaternion":
        return _matrix_to_quaternion(matrix)
    if to_rep == "rotation_6d":
        return _matrix_to_rotation_6d(matrix)
    if to_rep == "axis_angle":
        return _matrix_to_axis_angle(matrix)
    if to_rep == "euler_angles":
        if to_convention is None:
            raise ValueError("Euler-angle output requires to_convention.")
        return _matrix_to_euler_angles(matrix, to_convention)
    raise NotImplementedError(f"Rotation representation {to_rep!r} is not implemented.")


def rotation_transform(rot, from_rep, to_rep, from_convention=None, to_convention=None):
    """
    Transform a rotation representation into another equivalent rotation representation.
    """
    if from_rep not in VALID_ROTATION_REPRESENTATIONS:
        raise ValueError(f"Invalid rotation representation: {from_rep}")
    if to_rep not in VALID_ROTATION_REPRESENTATIONS:
        raise ValueError(f"Invalid rotation representation: {to_rep}")
    if from_rep == to_rep and from_convention == to_convention:
        return np.asarray(rot, dtype=np.float32)
    mat = _to_matrix(rot, from_rep, from_convention=from_convention)
    return _from_matrix(mat, to_rep, to_convention=to_convention)


def xyz_rot_transform(xyz_rot, from_rep, to_rep, from_convention=None, to_convention=None):
    """
    Transform an xyz_rot representation into another equivalent xyz_rot representation.
    """
    if from_rep not in VALID_ROTATION_REPRESENTATIONS:
        raise ValueError(f"Invalid rotation representation: {from_rep}")
    if to_rep not in VALID_ROTATION_REPRESENTATIONS:
        raise ValueError(f"Invalid rotation representation: {to_rep}")
    if from_rep == to_rep and from_convention == to_convention:
        return np.asarray(xyz_rot, dtype=np.float32)

    xyz_rot = np.asarray(xyz_rot, dtype=np.float32)
    if from_rep != "matrix":
        expected = 3 + ROTATION_REPRESENTATION_DIMS[from_rep]
        if xyz_rot.shape[-1] != expected:
            raise ValueError(f"Expected xyz_rot shape (..., {expected}), got {xyz_rot.shape}.")
        xyz = xyz_rot[..., :3]
        rot = xyz_rot[..., 3:]
    else:
        if xyz_rot.shape[-2:] != (4, 4):
            raise ValueError(f"Expected pose matrices with shape (..., 4, 4), got {xyz_rot.shape}.")
        xyz = xyz_rot[..., :3, 3]
        rot = xyz_rot[..., :3, :3]

    rot = rotation_transform(
        rot=rot,
        from_rep=from_rep,
        to_rep=to_rep,
        from_convention=from_convention,
        to_convention=to_convention,
    )
    if to_rep != "matrix":
        return np.concatenate((xyz, rot), axis=-1).astype(np.float32)

    res = np.zeros(xyz.shape[:-1] + (4, 4), dtype=np.float32)
    res[..., :3, :3] = rot
    res[..., :3, 3] = xyz
    res[..., 3, 3] = 1.0
    return res


def xyz_rot_to_mat(xyz_rot, rotation_rep, rotation_rep_convention=None):
    """
    Transform an xyz_rot representation under any rotation form to a unified 4x4 pose representation.
    """
    return xyz_rot_transform(
        xyz_rot,
        from_rep=rotation_rep,
        to_rep="matrix",
        from_convention=rotation_rep_convention,
    )


def mat_to_xyz_rot(mat, rotation_rep, rotation_rep_convention=None):
    """
    Transform a unified 4x4 pose representation to an xyz_rot representation under any rotation form.
    """
    return xyz_rot_transform(
        mat,
        from_rep="matrix",
        to_rep=rotation_rep,
        to_convention=rotation_rep_convention,
    )


def apply_mat_to_pose(pose, mat, rotation_rep, rotation_rep_convention=None):
    """
    Apply transformation matrix mat to pose under any rotation form.
    """
    if rotation_rep not in VALID_ROTATION_REPRESENTATIONS:
        raise ValueError(f"Invalid rotation representation: {rotation_rep}")
    mat = np.asarray(mat, dtype=np.float32)
    pose = np.asarray(pose, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected mat shape (4, 4), got {mat.shape}.")
    if rotation_rep == "matrix":
        if pose.shape[-2:] != (4, 4):
            raise ValueError(f"Expected pose shape (..., 4, 4), got {pose.shape}.")
        return mat @ pose
    expected = 3 + ROTATION_REPRESENTATION_DIMS[rotation_rep]
    if pose.shape[-1] != expected:
        raise ValueError(f"Expected pose shape (..., {expected}), got {pose.shape}.")
    pose_mat = xyz_rot_to_mat(
        xyz_rot=pose,
        rotation_rep=rotation_rep,
        rotation_rep_convention=rotation_rep_convention,
    )
    res_pose_mat = mat @ pose_mat
    return mat_to_xyz_rot(
        mat=res_pose_mat,
        rotation_rep=rotation_rep,
        rotation_rep_convention=rotation_rep_convention,
    )


def apply_mat_to_pcd(pcd, mat):
    """
    Apply transformation matrix mat to point cloud.
    """
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected mat shape (4, 4), got {mat.shape}.")
    pcd[..., :3] = (mat[:3, :3] @ pcd[..., :3].T).T + mat[:3, 3]
    return pcd


def rot_mat_x_axis(angle):
    """
    3x3 transformation matrix for rotation along x axis.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_mat_y_axis(angle):
    """
    3x3 transformation matrix for rotation along y axis.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=np.float32)


def rot_mat_z_axis(angle):
    """
    3x3 transformation matrix for rotation along z axis.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def rot_mat(angles):
    """
    3x3 transformation matrix for rotation along x, y, z axes.
    """
    x_mat = rot_mat_x_axis(angles[0])
    y_mat = rot_mat_y_axis(angles[1])
    z_mat = rot_mat_z_axis(angles[2])
    return z_mat @ y_mat @ x_mat


def trans_mat(offsets):
    """
    4x4 transformation matrix for translation along x, y, z axes.
    """
    res = np.identity(4, dtype=np.float32)
    res[:3, 3] = np.asarray(offsets, dtype=np.float32)
    return res


def rot_trans_mat(offsets, angles):
    """
    4x4 transformation matrix for rotation along x, y, z axes, then translation along x, y, z axes.
    """
    res = np.identity(4, dtype=np.float32)
    res[:3, :3] = rot_mat(angles)
    res[:3, 3] = np.asarray(offsets, dtype=np.float32)
    return res


def trans_rot_mat(offsets, angles):
    """
    4x4 transformation matrix for translation along x, y, z axes, then rotation along x, y, z axes.
    """
    res = np.identity(4, dtype=np.float32)
    res[:3, :3] = rot_mat(angles)
    offsets = np.asarray(offsets, dtype=np.float32)
    res[:3, 3] = res[:3, :3] @ offsets
    return res
