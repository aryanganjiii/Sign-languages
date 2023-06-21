import cv2
import mediapipe as mp
import numpy as np

def find_degree(base_point, second_point):
    dx = base_point[0] - second_point[0]
    dy = base_point[1] - second_point[1]
    theta = np.degrees(np.arctan2(dy, dx))
    theta = theta if theta > 0 else theta + 360
    return theta

def is_letter_D(hand_landmarks):
    if hand_landmarks[12][1] > hand_landmarks[10][1] and hand_landmarks[16][1] > hand_landmarks[14][1] and hand_landmarks[20][1] > hand_landmarks[18][1] and hand_landmarks[4][0] < hand_landmarks[2][0] and hand_landmarks[8][1] < hand_landmarks[7][1]:
        return True
    else:
        return False

def is_letter_i(hand_landmarks):
    if hand_landmarks[12][1] > hand_landmarks[11][1] and hand_landmarks[16][1] > hand_landmarks[14][1] and hand_landmarks[8][1] > hand_landmarks[7][1] and hand_landmarks[4][0] < hand_landmarks[2][0] and hand_landmarks[20][1] < hand_landmarks[19][1]:
        return True
    else:
        return False

def is_letter_N(hand_landmarks):
    if hand_landmarks[12][1] > hand_landmarks[10][1] and hand_landmarks[16][1] > hand_landmarks[14][1] and hand_landmarks[20][1] > hand_landmarks[18][1] and hand_landmarks[4][0] < hand_landmarks[2][0] and hand_landmarks[4][1] > hand_landmarks[6][1] and hand_landmarks[4][1] > hand_landmarks[10][1] :
        return True
    else:
        return False

def is_letter_T(hand_landmarks):
    if   hand_landmarks[12][1] > hand_landmarks[11][1] and hand_landmarks[16][1] and hand_landmarks[4][1] < hand_landmarks[6][1] and hand_landmarks[4][0] < hand_landmarks[3][0] :
        return True
    else:
        return False

def is_letter_Y(hand_landmarks):
    if hand_landmarks[12][1] > hand_landmarks[11][1] and hand_landmarks[16][1] > hand_landmarks[14][1] and hand_landmarks[8][1] > hand_landmarks[7][1] and hand_landmarks[4][0] > hand_landmarks[3][0]:
        return True
    else:
        return False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        finger_letter = ""
        degree = 90
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                handLandmarks = []
                for landmarks in hand_landmarks.landmark:
                    handLandmarks.append([landmarks.x, landmarks.y])
                
                if is_letter_D(handLandmarks):
                    finger_letter = "D"
                elif is_letter_i(handLandmarks):
                    finger_letter = "i"
                elif is_letter_N(handLandmarks):
                    finger_letter = "N"
                elif is_letter_T(handLandmarks):
                    finger_letter = "T"
                elif is_letter_Y(handLandmarks):
                    finger_letter = "Y"
                
                degree = int(find_degree(handLandmarks[0], handLandmarks[12]))

                dx = handLandmarks[0][0] - handLandmarks[9][0]
                dy = handLandmarks[0][1] - handLandmarks[9][1]

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.putText(image, f"{finger_letter} - {degree} degree", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)
        cv2.imshow('MediaPipe Image', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
