import cv2
import supervision as sv
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('Training/runs/detect/train7/weights/best.pt')  # Adjust the path if needed

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kameradan görüntü gelmiyor...")
    exit()

# Create annotators for bounding boxes and labels
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Set a confidence threshold (adjust as needed)
confidence_threshold = 0.5

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run inference on the current frame
    results = model(frame)

    # Extract the first result
    result = results[0]

    # Process detections
    detections = sv.Detections.from_ultralytics(result)

    # Filter out detections below the confidence threshold
    for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
        if confidence < confidence_threshold:
            continue  # Skip low-confidence detections

        # Annotate the bounding boxes and labels on the frame
        label = f"{model.names[class_id]}: {confidence:.2f}"  # Class name and confidence score
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the annotated image in the window
    cv2.imshow("Webcam", frame)

    # Press 'Esc' to exit
    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Esc tuşuna basıldı.. Kapatılıyor..")
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
