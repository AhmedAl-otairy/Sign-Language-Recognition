import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Use correct camera index + backend (important for Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if camera opens
if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

for j in range(number_of_classes):
    class_path = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print(f'Collecting data for class {j}')

    # Wait for user to press 'q'
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0

    # Capture images
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        img_path = os.path.join(class_path, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()