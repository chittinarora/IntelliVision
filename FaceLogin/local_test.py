import face_recognition
import cv2
import numpy as np

def capture_face():
    cap = cv2.VideoCapture(0)
    encoding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip if needed
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(rgb)
        if faces:
            encoding = face_recognition.face_encodings(rgb, faces)[0]

            # Draw a box
            top, right, bottom, left = faces[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Face Captured", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Capture Test", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q') or encoding is not None:
            break

    cap.release()
    cv2.destroyAllWindows()
    return encoding

if __name__ == "__main__":
    encoding = capture_face()
    if encoding is not None:
        print("\nFace encoding captured!")
        print("Encoding vector (128D):")
        print(encoding)
    else:
        print("No face detected.")
