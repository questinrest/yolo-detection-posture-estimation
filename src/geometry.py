import numpy as np
import math

def create_vectors(keypoints):
    vectors = []
    for i in range(len(keypoints) - 1):
        dx = keypoints[i + 1][0] - keypoints[i][0]
        dy = keypoints[i + 1][1] - keypoints[i][1]
        vectors.append(np.array([dx, dy]))
    return vectors

def joint_angle(v1, v2):
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)
    if mag == 0:
        return 0.0

    cos_angle = np.clip(np.dot(v1, v2) / mag, -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))
    return 180 - angle

def extract_angles(vectors):
    if len(vectors) < 3:
        return []

    hip_angle = joint_angle(vectors[0], vectors[1])
    knee_angle = joint_angle(vectors[1], vectors[2])

    return [hip_angle, knee_angle]

def vertical_distance(hip, knee):
    return abs(hip[1] - knee[1])
