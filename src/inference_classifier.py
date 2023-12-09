import pickle

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn 
import time 

# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(21*2,100)
        self.Matrix2 = nn.Linear(100,50)
        self.Matrix3 = nn.Linear(50,100)
        self.Matrix4 = nn.Linear(100,26)
        self.R = nn.ReLU()
    def forward(self,x):
        x = x.view(-1,21*2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.R(self.Matrix3(x))
        x = self.Matrix4(x)
        return x.squeeze()

model = torch.load("./model_0.975.pth")

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {i: chr(ord('A') + i) for i in range(0, 26)}

print(labels_dict)

predicted_character = ""
takeLetter = True

start = time.time()

while True:

    data_aux = []
    x_ = []
    y_ = []


    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        #prediction = model.predict([np.asarray(data_aux)])

        arr = torch.from_numpy(np.array([value 
                    for landmark in results.multi_hand_landmarks[0].landmark 
                    for value in (landmark.x, landmark.y)
                ])).float()


        #print(predicted_character)

        arr = torch.from_numpy(np.array([value 
                for landmark in results.multi_hand_landmarks[0].landmark 
                for value in (landmark.x, landmark.y)
            ])).float()

        predicted_character = labels_dict[int(model(arr).argmax())]

        current_time = int(time.time() - start)
        #print(current_time)
        if current_time % 5 == 0:  # Adjust the threshold for a more accurate capture
            if takeLetter:
                with open("./message.txt", "a") as f:
                    f.write(predicted_character)


            takeLetter = False 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            takeLetter = True 

        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
            

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
