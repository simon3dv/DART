import pickle
import sys,os
import numpy as np
import torch
import quat
from smplx.joint_names import JOINT_NAMES
from smplx import SMPLX
def axis_angle_to_matrix(axis_angle):
    """
    Convert axis-angle rotation to rotation matrix using quat.py functions.
    Args:
        axis_angle: numpy array of shape (*, 3)
    Returns:
        rotation_matrix: numpy array of shape (*, 3, 3)
    """
    # Convert axis-angle to quaternion first
    axis_angle = axis_angle
    quaternions = quat.from_axis_angle(axis_angle)
    # Convert quaternion to rotation matrix
    return quat.to_xform(quaternions)

def matrix_to_axis_angle(rotation_matrix):
    """
    Convert rotation matrix to axis-angle representation using quat.py functions.
    Args:
        rotation_matrix: numpy array of shape (*, 3, 3)
    Returns:
        axis_angle: numpy array of shape (*, 3)
    """
    # Convert rotation matrix to quaternion first
    rotation_matrix = rotation_matrix
    quaternions = quat.from_xform(rotation_matrix)
    # Convert quaternion to axis-angle
    return quat.to_scaled_angle_axis(quaternions)

def quaternion_to_matrix(quaternions):
    """
    Convert quaternions to rotation matrices using quat.py functions.
    Args:
        quaternions: numpy array of shape (*, 4) with real part first
    Returns:
        rotation_matrix: numpy array of shape (*, 3, 3)
    """
    return quat.to_xform(quaternions)


def qbetween(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v = torch.cross(v0, v1)
    w = torch.sqrt((v0 ** 2).sum(dim=-1, keepdim=True) * (v1 ** 2).sum(dim=-1, keepdim=True)) + (v0 * v1).sum(dim=-1,
                                                                                                              keepdim=True)
    return qnormalize(torch.cat([w, v], dim=-1))
def qnormalize(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    return q / torch.norm(q, dim=-1, keepdim=True)

def qbetween_np(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    return qbetween(v0, v1).numpy()

def qrot(q, v):
    '''
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    '''
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def face_z_transform(positions, global_orient, trans, deg=0):
    '''
    positions: [num_frame, num_joints, 3]
    global_orient: [num_frame, 3] axis-angle
    trans: [num_frame, 3]
    deg=0 mean Z+
    '''
    r_hip = JOINT_NAMES.index('right_hip')
    l_hip = JOINT_NAMES.index('left_hip')
    sdr_r = JOINT_NAMES.index('right_shoulder')
    sdr_l = JOINT_NAMES.index('left_shoulder')

    root_pos_init = positions[0]
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.linalg.norm(across)
    #across array([-0.9989424 , -0.02621989, -0.03777245], dtype=float32)
    #up = np.array([0, 1, 0])
    up = np.array([0, 0, 1])
    forward = np.cross(up, across)
    forward = forward / np.linalg.norm(forward)
    #print(f"forward={forward}")
    #forward array([-0.03778544,  0.        ,  0.99928588])
    theta = np.deg2rad(deg)
    target = np.array([np.sin(theta), -np.cos(theta), 0])
    #target = np.array([np.sin(theta), 0, np.cos(theta)]) 
    #print(target)
    #target = np.array([0, -1, 0])
    #target = np.array([0, 0, 1])
    #forward = torch.tensor(forward).unsqueeze(0).float()
    #target = torch.tensor(target).unsqueeze(0).float()
    q_rot = qbetween_np(forward, target)  # [1, 4]
    rotmat = quaternion_to_matrix(q_rot)  # [1, 3, 3]
     
    R_global = axis_angle_to_matrix(global_orient)  # [N, 3, 3]
    R_new = np.matmul(rotmat, R_global)
    global_orient_new = matrix_to_axis_angle(R_new)
    """
    trans = trans.copy()
    floor_y = positions[0, :, 1].min()
    trans[:, 1] -= floor_y

    root_xz = positions[0, 0] * np.array([1, 0, 1])
    trans -= root_xz

    trans_torch = torch.tensor(trans).float().cuda()
    q_rot_expanded = q_rot.expand(trans_torch.shape[0], -1)  # shape: [21, 4]
    trans_rot = qrot(q_rot_expanded, trans_torch)

    #trans_rot = qrot(q_rot, trans_torch)
    """
    return global_orient_new, trans#_rot.cpu().numpy()


if __name__ == "__main__":
    deg = float(sys.argv[-2])
    output_dir = sys.argv[-1]
    # Load input
    with open("data/stand.pkl", "rb") as f:
        data = pickle.load(f)
    N = data["global_orient"].shape[0]
    smplx_model = SMPLX('./data/smplx_lockedhead_20230207/models_lockedhead/smplx/SMPLX_NEUTRAL.npz', num_betas=10, model_type='smplx', flat_hand_mean=True, num_expression_coeffs=10, use_pca=False)#.cuda()
    smplx_output = smplx_model(betas=torch.tensor(data['betas']).float().view(1,10).repeat(N,1), body_pose=torch.tensor(data['body_pose']).float(),
            global_orient=torch.tensor(data['global_orient']).float(), pose2rot=True, jaw_pose=torch.zeros(N,3), leye_pose=torch.zeros(N,3), reye_pose=torch.zeros(N,3),
                               left_hand_pose=torch.zeros(N,45), right_hand_pose=torch.zeros(N,45),
                               expression=torch.zeros(N,10), transl=torch.tensor(data['transl']).float())

    vertices = smplx_output.vertices
    joints = smplx_output.joints
    positions = joints  # or change key if needed
    global_orient = data["global_orient"]
    transl = data["transl"] 

    # Transform to face Z+
    print(f"==debug==>positions.shape:{positions.shape}")
    print(f"==debug==>global_orient,.shape:{global_orient.shape}")
    print(f"==debug==>transl.shape:{transl.shape}")

    global_orient_new, transl_new = face_z_transform(positions.numpy(), global_orient, transl, deg=deg)
    print(f"==debug==>global_orient:{global_orient_new[0]}")
    print(f"==debug==>transl:{transl[0]}")

    # Save result
    data["global_orient"] = global_orient_new
    data["transl"] = transl_new
    #data["transl"][:,0]+=5
    print(f"Saved to data/stand_rot_init.pkl")
    os.makedirs(output_dir,exist_ok=True)
    with open(f"{output_dir}/sequence_init.pkl", "wb") as f:
        pickle.dump(data, f)




