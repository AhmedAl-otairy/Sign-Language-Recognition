import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# ===== Load model بشكل صحيح مهما كان مكان التشغيل =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.p')

model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

# ===== تشغيل الكاميرا =====
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Cannot open camera")
    exit()

# ===== إعداد MediaPipe =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3
)

labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# ===== التشغيل =====
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    # حماية من الخطأ
    if not ret or frame is None:
        continue

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            min_x, min_y = min(x_), min(y_)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # رسم النتائج
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.putText(frame, predicted_character,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('Sign Language Detector', frame)

    # خروج بزر q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== إغلاق =====
cap.release()
cv2.destroyAllWindows()