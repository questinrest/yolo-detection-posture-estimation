from .config import KNEE_STRAIGHT, KNEE_BENT, HIP_STRAIGHT, HIP_BENT, VERTICAL_DIST_SITTING

def logic_check(hip_angle, knee_angle, hip, knee):
    knee_straight = knee_angle > KNEE_STRAIGHT
    knee_bent = knee_angle < KNEE_BENT
    hip_straight = hip_angle > HIP_STRAIGHT
    hip_bent = hip_angle < HIP_BENT

    vertical_dist = abs(hip[1] - knee[1])

    if knee_straight and hip_straight:
        return "Standing"

    if knee_bent and hip_bent and vertical_dist < VERTICAL_DIST_SITTING:
        return "Sitting"

    if knee_straight and hip_bent:
        return "Bending"

    return "Unknown"
