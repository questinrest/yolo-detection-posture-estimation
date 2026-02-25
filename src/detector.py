import cv2
from .geometry import create_vectors, extract_angles
from .posture_logic import logic_check
from .config import CONFIDENCE_THRESHOLD, IOU_THRESHOLD

def detect_frame_wise(model, frame):
    image = frame.copy()
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

    class_ids = results[0].boxes.cls.int().tolist()
    conf_scores = [round(c, 2) for c in results[0].boxes.conf.tolist()]
    boxes = [
        (tuple(b[:2].int().tolist()), tuple(b[2:].int().tolist()))
        for b in results[0].boxes.xyxy
    ]

    for i in range(len(boxes)):
        label = f"{results[0].names[class_ids[i]]} {int(conf_scores[i]*100)}%"
        x1, y1 = boxes[i][0]
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )
        cv2.rectangle(image, boxes[i][0], boxes[i][1], (255, 0, 0), 1)

    return image


def process_image(model, image_path):
    image = cv2.imread(image_path)
    results = model(image_path, verbose=False)

    if not results or results[0].keypoints is None:
        return image, "No person detected"

    keypoints_all = results[0].keypoints.xy[0].tolist()

    right_idx = [6, 12, 14, 16]
    left_idx = [5, 11, 13, 15]

    right_pts = [keypoints_all[i] for i in right_idx]
    left_pts = [keypoints_all[i] for i in left_idx]

    right_vecs = create_vectors(right_pts)
    left_vecs = create_vectors(left_pts)

    right_angles = extract_angles(right_vecs)
    left_angles = extract_angles(left_vecs)

    pose = "Unknown"

    if len(right_angles) == 2:
        r_hip, r_knee = right_angles
        pose = logic_check(r_hip, r_knee, right_pts[1], right_pts[2])

    if pose == "Unknown" and len(left_angles) == 2:
        l_hip, l_knee = left_angles
        pose = logic_check(l_hip, l_knee, left_pts[1], left_pts[2])

    # ---- Draw bounding box ----
    box = results[0].boxes.xyxy[0].int().tolist()
    x1, y1, x2, y2 = box

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        image,
        pose,
        (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return image, pose
