import numpy as np

# Calculate cross object of two 3D vectors.
def _fast_cross(a, b):
    return np.concatenate([
        a[...,1:2]*b[...,2:3] - a[...,2:3]*b[...,1:2],
        a[...,2:3]*b[...,0:1] - a[...,0:1]*b[...,2:3],
        a[...,0:1]*b[...,1:2] - a[...,1:2]*b[...,0:1]], axis=-1)

# Make origin quaternions (No rotations)
def eye(shape, dtype=np.float32):
    return np.ones(list(shape) + [4], dtype=dtype) * np.asarray([1, 0, 0, 0], dtype=dtype)

# Return norm of quaternions
def length(x):
    return np.sqrt(np.sum(x * x, axis=-1))

# Make unit quaternions
def normalize(x, eps=1e-8):
    return x / (length(x)[...,None] + eps)

def abs(x):
    return np.where(x[...,0:1] > 0.0, x, -x)

# Calculate inverse rotations
def inv(q):
    return np.array([1, -1, -1, -1], dtype=np.float32) * q

# Calculate the dot product of two quaternions
def dot(x, y):
    return np.sum(x * y, axis=-1)[...,None] if x.ndim > 1 else np.sum(x * y, axis=-1)

# Multiply two quaternions (return rotations).
def mul(x, y):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

def inv_mul(x, y):
    return mul(inv(x), y)

def mul_inv(x, y):
    return mul(x, inv(y))

# Multiply quaternions and vectors (return vectors).
def mul_vec(q, x):
    t = 2.0 * _fast_cross(q[..., 1:], x)
    return x + q[..., 0][..., None] * t + _fast_cross(q[..., 1:], t)

def inv_mul_vec(q, x):
    return mul_vec(inv(q), x)

def unroll(x):
    y = x.copy()
    for i in range(1, len(x)):
        d0 = np.sum( y[i] * y[i-1], axis=-1)
        d1 = np.sum(-y[i] * y[i-1], axis=-1)
        y[i][d0 < d1] = -y[i][d0 < d1]
    return y
# Calculate quaternions between two 3D vectors (x to y).
def between(x, y):
    return np.concatenate([
        np.sqrt(np.sum(x*x, axis=-1) * np.sum(y*y, axis=-1))[...,None] +
        np.sum(x * y, axis=-1)[...,None],
        _fast_cross(x, y)], axis=-1)

def log(x, eps=1e-5):
    length = np.sqrt(np.sum(np.square(x[...,1:]), axis=-1))[...,None]
    halfangle = np.where(length < eps, np.ones_like(length), np.arctan2(length, x[...,0:1]) / length)
    return halfangle * x[...,1:]

def exp(x, eps=1e-5):
    halfangle = np.sqrt(np.sum(np.square(x), axis=-1))[...,None]
    c = np.where(halfangle < eps, np.ones_like(halfangle), np.cos(halfangle))
    s = np.where(halfangle < eps, np.ones_like(halfangle), np.sinc(halfangle / np.pi))
    return np.concatenate([c, s * x], axis=-1)

# Calculate global space rotations and positions from local space.
def fk(lrot, lpos, parents):

    gp, gr = [lpos[...,:1,:]], [lrot[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(mul    (gr[parents[i]], lrot[...,i:i+1,:]))

    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)

def fk_rot(lrot, parents):

    gr = [lrot[...,:1,:]]
    for i in range(1, len(parents)):
        gr.append(mul(gr[parents[i]], lrot[...,i:i+1,:]))

    return np.concatenate(gr, axis=-2)

# Calculate local space rotations and positions from global space.
def ik(grot, gpos, parents):

    return (
        np.concatenate([
            grot[...,:1,:],
            mul(inv(grot[...,parents[1:],:]), grot[...,1:,:]),
        ], axis=-2),
        np.concatenate([
            gpos[...,:1,:],
            mul_vec(
                inv(grot[...,parents[1:],:]),
                gpos[...,1:,:] - gpos[...,parents[1:],:]),
        ], axis=-2))

def ik_rot(grot, parents):

    return np.concatenate([grot[...,:1,:],
                        mul(inv(grot[...,parents[1:],:]), grot[...,1:,:]),
                    ], axis=-2)
def fk_vel(lrot, lpos, lvel, lang, parents):

    gp, gr, gv, ga = [lpos[...,:1,:]], [lrot[...,:1,:]], [lvel[...,:1,:]], [lang[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(mul    (gr[parents[i]], lrot[...,i:i+1,:]))
        gv.append(mul_vec(gr[parents[i]], lvel[...,i:i+1,:]) +
            _fast_cross(ga[parents[i]], mul_vec(gr[parents[i]], lpos[...,i:i+1,:])) +
            gv[parents[i]])
        ga.append(mul_vec(gr[parents[i]], lang[...,i:i+1,:]) + ga[parents[i]])

    return (
        np.concatenate(gr, axis=-2),
        np.concatenate(gp, axis=-2),
        np.concatenate(gv, axis=-2),
        np.concatenate(ga, axis=-2))

# Linear Interpolation of two vectors
def lerp(x, y, t):
    return (1 - t) * x + t * y

# LERP of quaternions
def quat_lerp(x, y, t):
    return normalize(lerp(x, y, t))

# Spherical linear interpolation of quaternions
def slerp(x, y, t):
    if t == 0:
        return x
    elif t == 1:
        return y

    if dot(x, y) < 0:
        y = - y
    ca = dot(x, y)
    theta = np.arccos(np.clip(ca, 0, 1))

    r = normalize(y - x * ca)

    return x * np.cos(theta * t) + r * np.sin(theta * t)


###################################################
# Calculate other rotations from other quaternions.
################################################### 
###################################################
# Calculate other rotations from other quaternions.
################################################### 

# Calculate euler angles from quaternions.
def to_euler(x, order='zyx'):

    q0 = x[...,0:1]
    q1 = x[...,1:2]
    q2 = x[...,2:3]
    q3 = x[...,3:4]

    if order == 'zyx':

        return np.concatenate([
            np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)),
            np.arcsin((2 * (q0 * q2 - q3 * q1)).clip(-1,1)),
            np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))], axis=-1)

    elif order == 'yzx':

        return np.concatenate([
            np.arctan2(2 * (q2 * q0 - q1 * q3),  q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0),
            np.arcsin((2 * (q1 * q2 + q3 * q0)).clip(-1,1)),
            np.arctan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)],axis=-1)

    elif order == 'zxy':

        return np.concatenate([
            np.arctan2(2 * (q0 * q3 - q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3),
            np.arcsin((2 * (q0 * q1 + q2 * q3)).clip(-1,1)),
            np.arctan2(2 * (q0 * q2 - q1 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)], axis=-1)

    elif order == 'yxz':

        return np.concatenate([
            np.arctan2(2 * (q1 * q3 + q0 * q2), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3),
            np.arcsin((2 * (q0 * q1 - q2 * q3)).clip(-1,1)),
            np.arctan2(2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)], axis=-1)

    else:
        raise NotImplementedError('Cannot convert from ordering %s' % order)

# Calculate rotation matrix from quaternions.
def to_xform(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2

    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)[...,None,:],
        np.concatenate([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)[...,None,:],
        np.concatenate([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)[...,None,:],
    ], axis=-2)

# Calculate 6d orthogonal rotation representation (ortho6d) from quaternions.
# https://github.com/papagina/RotationContinuity
def to_xform_xy(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2

    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz], axis=-1)[...,None,:],
        np.concatenate([xy + wz, 1.0 - (xx + zz)], axis=-1)[...,None,:],
        np.concatenate([xz - wy, yz + wx], axis=-1)[...,None,:],
    ], axis=-2)

# Calculate scaled angle axis from quaternions.
def to_scaled_angle_axis(x, eps=1e-5):
    return 2.0 * log(x, eps)


#############################################
# Calculate quaternions from other rotations.
#############################################

# Calculate quaternions from axis angles.
def from_angle_axis(angle, axis):
    c = np.cos(angle / 2.0)[..., None]
    s = np.sin(angle / 2.0)[..., None]
    q = np.concatenate([c, s * axis], axis=-1)
    return q

# Calculate quaternions from axis-angle.
#def from_axis_angle(rots):
#    angle = np.linalg.norm(rots, axis=-1)
#    axis = rots / angle[...,None]
#    return from_angle_axis(angle, axis)
def from_axis_angle(rots):
    """
    Convert rotation vectors (axis-angle representation) to quaternions.
    
    Parameters:
    -----------
    rots : numpy.ndarray
        Rotation vectors with shape (..., 3)
    
    Returns:
    --------
    numpy.ndarray
        Quaternions with shape (..., 4)
    """
    # Calculate the angle (magnitude of rotation vector)
    angle = np.linalg.norm(rots, axis=-1, keepdims=True)
    
    # Create a mask for zero or near-zero angles
    zero_mask = (angle < 1e-8)[..., 0]  # Remove extra dimension from keepdims
    
    # Initialize quaternions with identity for zero angles
    quats = np.zeros(rots.shape[:-1] + (4,))
    quats[..., 0] = 1.0  # w component = 1 for identity quaternion
    
    # Handle non-zero angles
    non_zero_mask = ~zero_mask
    if np.any(non_zero_mask):
        # Normalize the axis for non-zero angles
        axis = np.zeros_like(rots)
        axis[non_zero_mask] = rots[non_zero_mask] / angle[non_zero_mask]
        
        # Calculate quaternion components
        half_angle = angle / 2.0
        sin_half_angle = np.sin(half_angle)
        cos_half_angle = np.cos(half_angle)
        
        # Set non-zero quaternions (w, x, y, z)
        quats[non_zero_mask, 0] = cos_half_angle[non_zero_mask, 0]
        quats[non_zero_mask, 1:] = axis[non_zero_mask] * sin_half_angle[non_zero_mask]
    
    return quats

# Calculate quaternions from euler angles.
def from_euler(e, order='zyx'):
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = from_angle_axis(e[..., 0], axis[order[0]])
    q1 = from_angle_axis(e[..., 1], axis[order[1]])
    q2 = from_angle_axis(e[..., 2], axis[order[2]])

    return mul(q0, mul(q1, q2))

# Calculate quaternions from rotation matrix.
def from_xform(ts):

    return normalize(
        np.where((ts[...,2,2] < 0.0)[...,None],
            np.where((ts[...,0,0] >  ts[...,1,1])[...,None],
                np.concatenate([
                    (ts[...,2,1]-ts[...,1,2])[...,None],
                    (1.0 + ts[...,0,0] - ts[...,1,1] - ts[...,2,2])[...,None],
                    (ts[...,1,0]+ts[...,0,1])[...,None],
                    (ts[...,0,2]+ts[...,2,0])[...,None]], axis=-1),
                np.concatenate([
                    (ts[...,0,2]-ts[...,2,0])[...,None],
                    (ts[...,1,0]+ts[...,0,1])[...,None],
                    (1.0 - ts[...,0,0] + ts[...,1,1] - ts[...,2,2])[...,None],
                    (ts[...,2,1]+ts[...,1,2])[...,None]], axis=-1)),
            np.where((ts[...,0,0] < -ts[...,1,1])[...,None],
                np.concatenate([
                    (ts[...,1,0]-ts[...,0,1])[...,None],
                    (ts[...,0,2]+ts[...,2,0])[...,None],
                    (ts[...,2,1]+ts[...,1,2])[...,None],
                    (1.0 - ts[...,0,0] - ts[...,1,1] + ts[...,2,2])[...,None]], axis=-1),
                np.concatenate([
                    (1.0 + ts[...,0,0] + ts[...,1,1] + ts[...,2,2])[...,None],
                    (ts[...,2,1]-ts[...,1,2])[...,None],
                    (ts[...,0,2]-ts[...,2,0])[...,None],
                    (ts[...,1,0]-ts[...,0,1])[...,None]], axis=-1))))

# Calculate quaternions from ortho6d.
def from_xform_xy(x):

    c2 = _fast_cross(x[...,0], x[...,1])
    c2 = c2 / np.sqrt(np.sum(np.square(c2), axis=-1))[...,None]
    c1 = _fast_cross(c2, x[...,0])
    c1 = c1 / np.sqrt(np.sum(np.square(c1), axis=-1))[...,None]
    c0 = x[...,0]

    return from_xform(np.concatenate([
        c0[...,None],
        c1[...,None],
        c2[...,None]], axis=-1))

# Calculate quaternions from scaled angle axis.
def from_scaled_angle_axis(x, eps=1e-5):
    return exp(x / 2.0, eps)

def rotation_convert(rotation, from_type, to_type):
    assert from_type in ['axis_angle', 'quaternion', 'rotation_matrix']
    assert to_type in ['axis_angle', 'quaternion', 'rotation_matrix']
    if from_type == to_type:
        return rotation
    if from_type == 'axis_angle':
        rot = R.from_rotvec(rotation)
    elif from_type == 'quaternion':
        rot = R.from_quat(rotation)
    elif from_type == 'rotation_matrix':
        rot = R.from_matrix(rotation)
    if to_type == 'axis_angle':
        return rot.as_rotvec()
    elif to_type == 'quaternion':
        return rot.as_quat()
    elif to_type == 'rotation_matrix':
        return rot.as_matrix()

def quaternion_slerp(q1, q2, t):
    """
    球面线性插值（Slerp）函数
    :param q1: 第一个四元数，形状为 (4,)
    :param q2: 第二个四元数，形状为 (4,)
    :param t: 插值系数，范围在 [0, 1]
    :return: 插值后的四元数，形状为 (4,)
    """
    q1 = np.array(q1)
    q2 = np.array(q2)

    dot = np.dot(q1, q2)

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return (s1 * q1) + (s2 * q2)

def mean_of_quaternions(quaternions):
    """
    计算四元数的均值-瞎搞算法
    :param quaternions: 四元数列表，每个四元数形状为 (4,)
    :return: 均值四元数，形状为 (4,)
    """
    if len(quaternions) == 0:
        raise ValueError("Quaternion list is empty")

    mean_quaternion = quaternions[0]
    for i in range(1, len(quaternions)):
        mean_quaternion = quaternion_slerp(mean_quaternion, quaternions[i], 1 / (i + 1))

    return mean_quaternion

def quaternion_smoothing(quaternions, window_size=5):
    """
    对四元数序列进行消抖处理
    """
    smoothed_quaternions = []
    for i in range(len(quaternions)):
        start = max(0, i - window_size // 2)
        end = min(len(quaternions), i + window_size // 2 + 1)
        window_quaternions = quaternions[start:end]
        mean_quaternion = mean_of_quaternions(window_quaternions)
        mean_quaternion /= np.linalg.norm(mean_quaternion)  # 归一化
        smoothed_quaternions.append(mean_quaternion)
    return np.array(smoothed_quaternions)

def smooth_rotations(axis_angle_sequence, window_size, input_format="axis_angle", output_format="axis_angle"):
    """
    对轴角序列进行消抖处理
    :param axis_angle_sequence: 轴角序列，形状为 (N, *) 的数组
    :param window_size: 消抖窗口大小
    :param input_format: 输入数据的格式，可选 'axis_angle', 'quaternion', 或 'rotation_matrix'
    :param output_format: 输出数据的格式，可选 'axis_angle', 'quaternion', 或 'rotation_matrix'
    """
    quaternions = [rotation_convert(axis_angle, from_type=input_format, to_type="quaternion") for axis_angle in axis_angle_sequence]
    quaternions = quaternion_smoothing(quaternions, window_size=window_size)
    return [rotation_convert(quaternion, from_type="quaternion", to_type=output_format) for quaternion in quaternions]


def normalize_quaternions(all_body_quat):
    """
    Normalize quaternions and handle zero-length quaternions.
    
    Parameters:
    -----------
    all_body_quat : numpy.ndarray
        Quaternion data with shape (..., 4)
    
    Returns:
    --------
    numpy.ndarray
        Normalized quaternion data with the same shape as input
    """
    # Make a copy to avoid modifying the input
    quat_normalized = all_body_quat.copy()

    # Calculate norms of quaternions
    norms = np.linalg.norm(quat_normalized, axis=-1, keepdims=True)

    # Identify zero quaternions
    zero_mask = (norms == 0)[..., 0]  # Remove extra dimension from keepdims

    # Replace zero quaternions with identity quaternion [1, 0, 0, 0]
    quat_normalized[zero_mask] = np.array([1, 0, 0, 0])

    # Normalize all quaternions
    quat_normalized = quat_normalized / np.linalg.norm(quat_normalized, axis=-1, keepdims=True)
    return quat_normalized
def interpolate_transl(all_body_transl, fps, ex_fps):
    """
    Interpolate translation data to a new frame rate using linear interpolation.
    
    Parameters:
    -----------
    all_body_transl : numpy.ndarray
        Translation data with shape (frames, 3)
    fps : float
        Original frames per second
    ex_fps : float
        Target frames per second for interpolation
    
    Returns:
    --------
    numpy.ndarray
        Interpolated translation data with shape (ex_frames, 3)
    """
    duration = len(all_body_transl) / fps
    ex_frame = int(duration * ex_fps)

    # Initialize output array with the same shape as input, but with ex_frame rows
    ex_all_body_transl = np.zeros((ex_frame, 3))

    for i in range(ex_frame):
        original_time = i / ex_fps
        original_frame_float = original_time * fps

        # Get the indices of the two nearest frames
        frame_before = int(np.floor(original_frame_float))
        frame_after = int(np.ceil(original_frame_float))

        # Handle edge case for the last frame
        if frame_after >= len(all_body_transl):
            frame_after = len(all_body_transl) - 1

        # If the frames are the same, no interpolation needed
        if frame_before == frame_after:
            ex_all_body_transl[i] = all_body_transl[frame_before]
        else:
            # Calculate the interpolation weight
            weight = original_frame_float - frame_before

            # For translations, simple linear interpolation is appropriate
            t1 = all_body_transl[frame_before]
            t2 = all_body_transl[frame_after]

            # Linear interpolation
            ex_all_body_transl[i] = (1 - weight) * t1 + weight * t2

    return ex_all_body_transl

def interpolate_transl(all_body_transl, fps, ex_fps):
    """
    Interpolate translation data to a new frame rate using linear interpolation.

    Parameters:
    -----------
    all_body_transl : numpy.ndarray
        Translation data with shape (frames, 3)
    fps : float
        Original frames per second
    ex_fps : float
        Target frames per second for interpolation

    Returns:
    --------
    numpy.ndarray
        Interpolated translation data with shape (ex_frames, 3)
    """
    duration = len(all_body_transl) / fps
    ex_frame = int(duration * ex_fps)

    # Initialize output array with the same shape as input, but with ex_frame rows
    ex_all_body_transl = np.zeros((ex_frame, 3))

    for i in range(ex_frame):
        original_time = i / ex_fps
        original_frame_float = original_time * fps

        # Get the indices of the two nearest frames
        frame_before = int(np.floor(original_frame_float))
        frame_after = int(np.ceil(original_frame_float))

        # Handle edge case for the last frame
        if frame_after >= len(all_body_transl):
            frame_after = len(all_body_transl) - 1

        # If the frames are the same, no interpolation needed
        if frame_before == frame_after:
            ex_all_body_transl[i] = all_body_transl[frame_before]
        else:
            # Calculate the interpolation weight
            weight = original_frame_float - frame_before

            # For translations, simple linear interpolation is appropriate
            t1 = all_body_transl[frame_before]
            t2 = all_body_transl[frame_after]

            # Linear interpolation
            ex_all_body_transl[i] = (1 - weight) * t1 + weight * t2

    return ex_all_body_transl
def normalize_quaternions(all_body_quat):
    """
    Normalize quaternions and handle zero-length quaternions.

    Parameters:
    -----------
    all_body_quat : numpy.ndarray
        Quaternion data with shape (..., 4)

    Returns:
    --------
    numpy.ndarray
        Normalized quaternion data with the same shape as input
    """
    # Make a copy to avoid modifying the input
    quat_normalized = all_body_quat.copy()

    # Calculate norms of quaternions
    norms = np.linalg.norm(quat_normalized, axis=-1, keepdims=True)

    # Identify zero quaternions
    zero_mask = (norms == 0)[..., 0]  # Remove extra dimension from keepdims

    # Replace zero quaternions with identity quaternion [1, 0, 0, 0]
    quat_normalized[zero_mask] = np.array([1, 0, 0, 0])

    # Normalize all quaternions
    quat_normalized = quat_normalized / np.linalg.norm(quat_normalized, axis=-1, keepdims=True)
    return quat_normalized

def interpolate_quaternions(all_body_quat, fps, ex_fps):
    """
    Interpolate quaternion data to a new frame rate using linear interpolation.
    
    Parameters:
    -----------
    all_body_quat : numpy.ndarray
        Quaternion data with shape (frames, joints, 4)
    fps : float
        Original frames per second
    ex_fps : float
        Target frames per second for interpolation
    
    Returns:
    --------
    numpy.ndarray
        Interpolated quaternion data with shape (ex_frames, joints, 4)
    """
    duration = len(all_body_quat) / fps
    ex_frame = int(duration * ex_fps)
    num_joints = all_body_quat.shape[1]
    ex_all_body_quat = np.zeros((ex_frame, num_joints, 4))

    for i in range(ex_frame):
        original_time = i / ex_fps
        original_frame_float = original_time * fps
        # Get the indices of the two nearest frames
        frame_before = int(np.floor(original_frame_float))
        frame_after = int(np.ceil(original_frame_float))
        # Handle edge case for the last frame
        if frame_after >= len(all_body_quat):
            frame_after = len(all_body_quat) - 1
        # If the frames are the same, no interpolation needed
        if frame_before == frame_after:
            ex_all_body_quat[i] = all_body_quat[frame_before]
        else:
            # Calculate the interpolation weight
            weight = original_frame_float - frame_before
            # For quaternions, we should use spherical linear interpolation (slerp)
            # But for simplicity, here's a linear interpolation first
            for j in range(num_joints):
                q1 = all_body_quat[frame_before, j]
                q2 = all_body_quat[frame_after, j]
                # Ensure we're interpolating along the shortest path
                dot_product = np.sum(q1 * q2)
                if dot_product < 0:
                    q2 = -q2  # Flip the quaternion
                # Linear interpolation of quaternions (note: this doesn't preserve unit length)
                interp_quat = (1 - weight) * q1 + weight * q2
                # Normalize to ensure unit quaternion
                ex_all_body_quat[i, j] = interp_quat / np.linalg.norm(interp_quat)
    return ex_all_body_quat


