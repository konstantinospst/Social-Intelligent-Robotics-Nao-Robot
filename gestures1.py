import cv2
import mediapipe as mp
from sic_framework.core.message_python2 import CompressedImageMessage
from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_camera import NaoqiCameraConf
import queue
import numpy as np
import json
import time


from sic_framework.devices import Nao
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.services.dialogflow.dialogflow import (DialogflowConf, GetIntentRequest, RecognitionResult,
                                                          QueryResult, Dialogflow)
from sic_framework.devices.common_naoqi.naoqi_leds import NaoLEDRequest, NaoFadeRGBRequest
from sic_framework.devices.common_naoqi.naoqi_motion import (NaoqiAnimationRequest, NaoqiIdlePostureRequest, NaoPostureRequest,
                                                            NaoqiBreathingRequest)
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoBackgroundMovingRequest, NaoBasicAwarenessRequest


# Initial call required for the main function
imgs = queue.Queue()

# We will use mediapipe's face mesh and face detection.
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Threshold for lip distance fluctuations for each user
user_threshold1 = 20

# Main function
"""
WE WILL HAVE TO DOCUMENT IT. THEY WILL GRADE US BASED ON QUALITY OF THE CODE (20%)
INCLUDING EASE OF INTERPRETATION FOR EACH THING WE DO (COMMENTS, VARIABLE NAMES ETC.)
"""
def detector(image, check=False):
    # Face Detection initialization

    # image=np.flipud(image)
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection, \
            mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True) as face_mesh:

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

                cv2.putText(image, user_id, (text_x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                            cv2.LINE_AA)

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
                        cv2.circle(image, (x1, y1), 1, (100, 100, 0), -1)
                        cv2.circle(image, (x2, y2), 1, (100, 100, 0), -1)
                        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        user_talking = []
        for user_id, lip_distance in user_distances.items():
            if lip_distance > user_threshold1:
                print(f"{user_id} is talking!")
                user_talking.append("true")
            # else:
            #     print(f"{user_id} is not talking.")

        if len(user_talking) >= 2 and not check:
            print("There is an interruption")
            # Setting Eye LEDs to red
            reply = nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 0, 0, 2))
            check = True
            nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/IDontKnow_1"))
            nao.tts.request(NaoqiTextToSpeechRequest("I think there may have been an interruption, could you repeat?"))
        # if [user_distances[0]]> user_threshold1 and [user_distances[1]] > user_threshold1:

    return image

# Release resources
def on_image(image_message: CompressedImageMessage):
    # we could use cv2.imshow here, but that does not work on Mac OSX
    imgs.put(image_message.image)

# Connecting to the NAO and camera
nao = Nao(ip="10.0.0.44")
nao.top_camera.register_callback(on_image)

# Setting Nao's eyes LEDs to blue
reply = nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 0, 1, 0))

# Set the robot to be standing (should already be the case, just to make sure) and start the breathing behavior
nao.motion.request(NaoPostureRequest("Stand", .5))
nao.autonomous.request(NaoqiBreathingRequest("Body", True))

# Initialize the main interaction variables
interaction = True   # Boolean to break out of the main loop and end interaction

# We will need to keep track of time because our main loop is a continous call to the main function
# i.e. we call the function for every frame, so we need to be specially careful with our calls to any service.
start_time = time.time()
current_time = time.time()
total_time = current_time - start_time

# Main loop
while interaction:
    img = imgs.get()
    img = cv2.flip(img, 0)
    img_total = detector(img)
    cv2.imshow('', img_total[..., ::-1])
    cv2.waitKey(1)

# Once the interaction is finished, terminate some of NAO's processes
reply = nao.leds.request(NaoLEDRequest("FaceLeds", False))