import cv2
import mediapipe as mp
import pygame
import time
import subprocess

mp_face_detection = mp.solutions.face_detection #Shortening the face detection model name
mp_hands = mp.solutions.hands # shortening the hand detection model name

#Helper functions that draw the lines and dots around the faces and hands
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles


cap = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexis=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands, \
                mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

as face_detection:
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
                











    





