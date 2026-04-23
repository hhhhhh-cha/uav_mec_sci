# 这个文件负责把环境 action dict 转成 replay buffer 里能存的平坦向量
import numpy as np


def flatten_high_action(action_dict):
    """
    Flatten env-compatible action dict into a 1D vector.

    Layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_beta(K*M*M) ]
    """
    move_dist = np.asarray(action_dict["move_dist"], dtype=np.float32).reshape(-1)
    move_angle = np.asarray(action_dict["move_angle"], dtype=np.float32).reshape(-1)
    offload_ratio = np.asarray(action_dict["offload_ratio"], dtype=np.float32).reshape(-1)
    sched_beta = np.asarray(action_dict["sched_beta"], dtype=np.float32).reshape(-1)

    return np.concatenate(
        [move_dist, move_angle, offload_ratio, sched_beta],
        axis=0,
    ).astype(np.float32)


def get_flattened_action_dim(state):
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M * M