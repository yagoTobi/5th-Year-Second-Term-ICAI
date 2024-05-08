import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('model.h5', 'rb'))
model = model_dict['model']

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8)

# * Dictionary to save our 36 classes
categories = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "a",
    11: "b",
    12: "c",
    13: "d",
    14: "e",
    15: "f",
    16: "g",
    17: "h",
    18: "i",
    19: "j",
    20: "k",
    21: "l",
    22: "m",
    23: "n",
    24: "o",
    25: "p",
    26: "q",
    27: "r",
    28: "s",
    29: "t",
    30: "u",
    31: "v",
    32: "w",
    33: "x",
    34: "y",
    35: "z",
}

while True:
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min = W, H
            x_max, y_max = 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * W), int(landmark.y * H)
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

            # Extract hand bounding box with some padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(W, x_max + padding)
            y_max = min(H, y_max + padding)

            if x_max > x_min and y_max > y_min:  # Verificar que la región de la mano tiene dimensiones válidas
                # Extract hand region from frame
                hand_img = frame[y_min:y_max, x_min:x_max]

                # Resize hand region to match model input size
                hand_img_resized = cv2.resize(hand_img, (150, 150))

                # Make prediction using the model
                prediction = model.predict(np.expand_dims(hand_img_resized, axis=0), verbose=0)

                # Get predicted character label
                predicted_character = np.argmax(prediction, axis=1)

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (10, 10), (140, 40), (0, 0, 0), -1)
                cv2.putText(frame, categories[predicted_character[0]], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display hand region for debugging
                cv2.imshow('Hand Region', hand_img_resized)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
