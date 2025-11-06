import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
        
                # Normalize: subtract minimum to make position-invariant
                min_x, min_y = min(x_coords), min(y_coords)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x - min_x
                    y = hand_landmarks.landmark[i].y - min_y

                    # Create array with all landmark x and y coordinates
                    data_aux.append(x)
                    data_aux.append(y)

            # The data_aux array represents one image
            # The label is the directory name
            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

