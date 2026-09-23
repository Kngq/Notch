import cv2
import mediapipe as mp
import time
import subprocess

def show_touch_popup():
    subprocess.run([
        "osascript", "-e",
        'display notification "STOP TOUCHING YOUR FACE!" with title "TOUCH" sound name "ping"'
    ])


#Face and hand detection mediapipe models
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands


#Functions to draw borders around landmarks(hands, and face)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
prev_overlap = False

with mp_hands.Hands(model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands, \
                mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                    while cap.isOpened():
                        success, image = cap.read()
                        if not success:
                            print("Empty camera frame.")
                            continue
                        h, w, _ = image.shape

                        image.flags.writeable = False
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        hand_results = hands.process(image)
                        face_results = face_detection.process(image)

                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
                        face_box = None
                        if face_results.detections:
                            for detection in face_results.detections:
                                mp_drawing.draw_detection(image, detection)
                                bbox = detection.location_data.relative_bounding_box
                                
                                xmin = int(bbox.xmin * w)
                                ymin = int(bbox.ymin * h)
                                box_w = int(bbox.width * w)
                                box_h = int(bbox.height * h)
                                face_box = (xmin, ymin, xmin + box_w, ymin + box_h)
                                
                        overlap = False
                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                        image,
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())
                                if face_box is not None:
                                    fx1, fy1, fx2, fy2 = face_box
                                    for lm in hand_landmarks.landmark:
                                        px, py = int(lm.x * w), int(lm.y * h)
                                        if fx1 <= px <= fx2 and fy1 <= py <= fy2:
                                            overlap = True
                                            break

                        if overlap and not prev_overlap:
                             show_touch_popup()
                        prev_overlap = overlap

        
                        cv2.imshow('Mediapipe hands & face overlap', image)
                        if cv2.waitKey(5) & 0xFF == 27:
                            break

cap.release()
cv2.destroyAllWindows()

