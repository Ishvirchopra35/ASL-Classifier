import cv2
import pickle
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
               8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
               15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 
               22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Only process the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        x_coords = []
        y_coords = []
        
        # Collect original coordinates for bounding box
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_coords.append(x)
            y_coords.append(y)
        
        # Normalize for prediction
        min_x, min_y = min(x_coords), min(y_coords)
        for i in range(len(hand_landmarks.landmark)):
            normalized_x = x_coords[i] - min_x
            normalized_y = y_coords[i] - min_y
            data_aux.append(normalized_x)
            data_aux.append(normalized_y)

        # Use original coordinates for bounding box
        x1 = int(min(x_coords) * W) - 10
        y1 = int(min(y_coords) * H) - 10
        x2 = int(max(x_coords) * W) - 10
        y2 = int(max(y_coords) * H) - 10

        pred = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(pred[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)


    cv2.imshow('frame', frame)
    key = cv2.waitKey(25)
    if key == ord('q') or key == 27:  # 27 is ESC key
        break
    if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()