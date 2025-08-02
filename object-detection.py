import cv2
import torch
from ultralytics import YOLO

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Load and move model to device
    model = YOLO('yolov9c.pt').to(device)

    # Use half precision if GPU is available
    if device.type == 'cuda':
        model.half()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Resize frame for speed (optional: adjust resolution as needed)
        frame = cv2.resize(frame, (640, 480))

        # Convert to half precision if using GPU + half
        if device.type == 'cuda':
            frame = frame.astype('float16')

        # Run model with streaming
        results = model(frame, stream=True)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                label_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                class_label = model.names[label_id]
                label_text = f"{class_label}: {confidence:.2f}"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show the frame
        cv2.imshow("YOLOv9 Real-Time Detection", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
