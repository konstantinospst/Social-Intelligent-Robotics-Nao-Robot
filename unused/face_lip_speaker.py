import cv2
import mediapipe as mp
import numpy as np
import json

from sic_framework.devices import Nao
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.services.dialogflow.dialogflow import (DialogflowConf, GetIntentRequest, RecognitionResult,
                                                          QueryResult, Dialogflow)
from sic_framework.devices.common_naoqi.naoqi_leds import NaoLEDRequest, NaoFadeRGBRequest

# the callback function
def on_dialog(message):
    if message.response:
        if message.response.recognition_result.is_final:
            print("Transcript:", message.response.recognition_result.transcript)

# connect to the robot
nao = Nao(ip='192.168.178.45')

# load the key json file
keyfile_json = json.load(open("sictest-404013-2a1d5afc360b.json"))

# set up the config
conf = DialogflowConf(keyfile_json=keyfile_json, sample_rate_hertz=16000)

# initiate Dialogflow object
dialogflow = Dialogflow(ip='localhost', conf=conf)

# connect the output of NaoqiMicrophone as the input of DialogflowComponent
dialogflow.connect(nao.mic)

# register a callback function to act upon arrival of recognition_result
dialogflow.register_callback(on_dialog)

# Demo starts
nao.tts.request(NaoqiTextToSpeechRequest("Hello, who are you?"))
print(" -- Ready -- ")
x = np.random.randint(10000)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
cap = cv2.VideoCapture(0)
# Threshold for lip distance fluctuations for each user
user_threshold1 = 50

# Face Detection initialization
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection, \
        mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True) as face_mesh:
    print(" ----- Conversation turn -----")

    # Setting Nao's eyes LEDs to blue
    reply = nao.leds.request(NaoLEDRequest("FaceLeds", True))
    reply = nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0, 0, 1, 0))
    reply = nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0, 0, 1, 0))

    reply = dialogflow.request(GetIntentRequest(x))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detection.process(image_rgb)
        user_distances = {}  # Store lip distances for each user

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Get face position
                center_x = x + w // 2

                # Display ID on top of the bounding box
                user_id = "User 0" if center_x < iw // 2 else "User 1"
                text_size = cv2.getTextSize(user_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x if center_x < iw // 2 else x + w - text_size[0] - 5

                cv2.putText(image, user_id, (text_x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                # Process face mesh
                height, width, _ = image.shape
                results2 = face_mesh.process(image_rgb)
                    
                if results2.multi_face_landmarks:
                    for face_id, face_landmarks in enumerate(results2.multi_face_landmarks):
                        pt1 = face_landmarks.landmark[0]
                        x1 = int(pt1.x * width)
                        y1 = int(pt1.y * height)
                        pt2 = face_landmarks.landmark[17]
                        x2 = int(pt2.x * width)
                        y2 = int(pt2.y * height)
                        
                        # Calculate lip distance for each face
                        lip_distance = abs(y2 - y1)
                        user_distances[face_id] = lip_distance
                        
                        # Draw landmarks and lines on each face if required
                        cv2.circle(image, (x1, y1), 2, (100, 100, 0), -1)
                        cv2.circle(image, (x2, y2), 2, (100, 100, 0), -1)
                        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Perform actions based on lip distances for each user (user_distances)
        print(user_distances)
        user_talking = []
        for user_id, lip_distance in user_distances.items():
            if lip_distance > user_threshold1:
                print(f"{user_id} is talking!")
                user_talking.append("true")
            else:
                print(f"{user_id} is not talking.")

        if len(user_talking) >= 2:
            print("There is an interruption")
            # Setting right Eye LEDs to red
            reply = nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 1, 0, 0, 0))
            reply = nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 1, 0, 0, 0))
            nao.tts.request(NaoqiTextToSpeechRequest("I think there may have been an interruption, could you repeat?"))
        else:
            print(reply.intent)

            if reply.fulfillment_message:
                text = reply.fulfillment_message
                print("Reply:", text)
                nao.tts.request(NaoqiTextToSpeechRequest(text))

        # Display the image
        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
