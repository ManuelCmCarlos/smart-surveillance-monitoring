from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Horizontal counting line
line_y = 250

# Protected zone coordinates
zone_x1 = 200
zone_y1 = 120
zone_x2 = 500
zone_y2 = 400

# Store previous Y positions
previous_positions = {}

# Store IDs already counted
counted_ids = set()

# Store intrusion IDs
intrusion_ids = set()

# Entry / Exit counters
entry_count = 0
exit_count = 0

# Tracking history
track_history = defaultdict(list)

# Log file name
log_file = "events.log"

# Main loop
while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(frame, persist=True)

    annotated_frame = frame.copy()

    # Draw counting line
    cv2.line(
        annotated_frame,
        (0, line_y),
        (640, line_y),
        (255, 0, 255),
        3
    )

    # Draw protected zone
    cv2.rectangle(
        annotated_frame,
        (zone_x1, zone_y1),
        (zone_x2, zone_y2),
        (255, 0, 255),
        2
    )

    intrusion_detected = False

    for result in results:

        boxes = result.boxes

        if boxes.id is None:
            continue

        track_ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):

            class_id = int(box.cls[0])

            if class_id == 0:

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Save track history
                track_history[track_id].append(
                    (center_x, center_y)
                )

                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                # Draw box
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw center point
                cv2.circle(
                    annotated_frame,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1
                )

                # Draw trail
                points = track_history[track_id]

                for i in range(1, len(points)):
                    cv2.line(
                        annotated_frame,
                        points[i - 1],
                        points[i],
                        (0, 255, 255),
                        2
                    )

                # Label
                label = f"ID {track_id} | {confidence:.2f}"

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # Entry / Exit logic
                if track_id in previous_positions:

                    previous_y = previous_positions[track_id]

                    # ENTRY
                    if (
                        previous_y < line_y
                        and center_y >= line_y
                        and track_id not in counted_ids
                    ):
                        entry_count += 1
                        counted_ids.add(track_id)

                        with open(log_file, "a") as file:
                            file.write(
                                f"ENTRY - ID {track_id}\n"
                            )

                    # EXIT
                    elif (
                        previous_y > line_y
                        and center_y <= line_y
                        and track_id not in counted_ids
                    ):
                        exit_count += 1
                        counted_ids.add(track_id)

                        with open(log_file, "a") as file:
                            file.write(
                                f"EXIT - ID {track_id}\n"
                            )

                previous_positions[track_id] = center_y

                # Intrusion detection
                if (
                    zone_x1 < center_x < zone_x2
                    and
                    zone_y1 < center_y < zone_y2
                ):

                    intrusion_detected = True

                    if track_id not in intrusion_ids:

                        intrusion_ids.add(track_id)

                        with open(log_file, "a") as file:
                            file.write(
                                f"INTRUSION - ID {track_id}\n"
                            )

    # Display counters
    cv2.putText(
        annotated_frame,
        f"Entries: {entry_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Exits: {exit_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Intrusions: {len(intrusion_ids)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    if intrusion_detected:

        cv2.putText(
            annotated_frame,
            "INTRUSION DETECTED",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow(
        "Intrusion Monitoring System",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()