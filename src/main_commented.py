# Importing required libraries
import queue
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import threading

# Importing detector function and classes from the sic_framework
from detector import *
from sic_framework.devices import Nao
from sic_framework.devices.desktop import Desktop
from sic_framework.core.message_python2 import CompressedImageMessage
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest, NaoqiBreathingRequest, NaoqiAnimationRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoBasicAwarenessRequest
from sic_framework.services.dialogflow.dialogflow import (DialogflowConf, GetIntentRequest, RecognitionResult,
                                                          QueryResult, Dialogflow)

# Function to handle incoming dialog from Dialogflow
def on_dialog(message):
    if message.response:
        if message.response.recognition_result.is_final:
            print("Transcript:", message.response.recognition_result.transcript)

# Load Dialogflow configuration
keyfile_json = json.load(open("sictest-404013-2a1d5afc360b.json"))
conf = DialogflowConf(keyfile_json=keyfile_json, sample_rate_hertz=44100, language="en")
dialogflow = Dialogflow(ip='localhost', conf=conf)

# Function to enqueue incoming images for processing
def on_image(image_message: CompressedImageMessage):
    imgs.put(image_message.image)

# Queue for Dialogflow requests
dialogflow_request_queue = queue.Queue()

# Function to handle Dialogflow requests in a separate thread
def dialogflow_request_handler():
    # Demo starts
    nao.tts.request(NaoqiTextToSpeechRequest("Yes! I am listening"))

    while True:
        if not dialogflow_request_queue.empty():
            print(" ----- Conversation turn -----")
            x = dialogflow_request_queue.get()
            reply = dialogflow.request(GetIntentRequest(x))

            print(reply.intent)
            if reply.intent == "user name":
                text = reply.fulfillment_message
                print("Reply:", text)
                nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Hey_3"))
                nao.tts.request(NaoqiTextToSpeechRequest(text, animated=True))
            elif reply.intent == "Default Goodbye Intent":
                text = reply.fulfillment_message
                print("Reply:", text)
                nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Hey_3"))
                nao.tts.request(NaoqiTextToSpeechRequest(text, animated=True))
            elif reply.fulfillment_message:
                text = reply.fulfillment_message
                print("Reply:", text)
                nao.tts.request(NaoqiTextToSpeechRequest(text, animated=True))


# Initialize NAO robot and configure Dialogflow connection
nao = Nao(ip="10.0.0.121")
desktop = Desktop()
dialogflow.connect(desktop.mic)
dialogflow.register_callback(on_dialog)
x = np.random.randint(10000)

# Start the Dialogflow thread
dialogflow_thread = threading.Thread(target=dialogflow_request_handler)
dialogflow_thread.start()

# Set NAO robot posture, LED color, start breathing, and enable autonomous behavior
nao.motion.request(NaoPostureRequest("Stand", 1))
reply = nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 0, 1, 1))
nao.motion.request(NaoqiBreathingRequest("Body", True))
nao.autonomous.request(NaoBasicAwarenessRequest(True))

# Initialize variables for lip detection threshold and conversation state tracking
user_threshold = 10
start_time = 0
silence_tracking = 0
interruption_started = 0
silence_threshhold = 0.3
imgs = queue.Queue()
nao.top_camera.register_callback(on_image)

# Main loop for image processing and conversation management
while True:
    # Reset eyes to blue (when maajor interruption, they are set to red)
    reply = nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 0, 1, 0))
    img = imgs.get()
    img = cv2.flip(img, 0)
    img_total = detector(img)
    user_distances = img_total[1]

    # Determine if users are talking and handle Dialogflow requests
    user_talking = []
    for user_id, lip_distance in user_distances.items():
        if lip_distance > user_threshold:
            user_talking.append("true")

    # Handle Dialogflow requests and interruptions
        # Initialized when the 2 users are speaking
        # Also entered when one of them has not been silent for long - this is to avoid losing track of the interaction
    # Kind of too hard to fully explain here but 2 main tracking features:
       # elapsed_time: for major interruption handling
       # slience_tracking: avoids losing track of the interruption whenever the lips of the users are closed while speaking
    if (len(user_talking) == 1 and silence_tracking <= silence_threshhold) or (len(user_talking) == 2):
        dialogflow_request_queue.put(x)  # Put request in the Dialogflow queue
        if len(user_talking) >= 2:
            silence_tracking = 0  # Reset silence tracking when both users are talking
            # Update start_time to track the duration of simultaneous talking
            if start_time == 0:
                start_time = time.time()  # Record start time of the interruption
            else:
                current_time = time.time()
                elapsed_time = current_time - start_time + silence_tracking
                # Handle interruptions lasting longer than 3 seconds (major interruptions)
                if elapsed_time >= 3:
                    print("There is an interruption")
                    start_time = 0 # Reset start_time after handling interruption

                    # Change LED color to red to indicate interruption
                    reply = nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 0, 0, 0))
                    nao.tts.request(NaoqiTextToSpeechRequest(
                        "I think there may have been an interruption. When the eyes turn blue, please try again?",
                        animated=True))
                    time.sleep(3)
        elif silence_tracking > silence_threshhold:
            # Reset tracking variables if silence exceeds threshold (this means there is no interruption anymore)
            start_time = 0
            silence_tracking = 0
        else:
            # Track the start of silence if one of the users is not talking
            # This is really important to avoid reseting the elapsed time
            if interruption_started == 0:
                interruption_started = time.time()
            else:
                # Update silence tracking duration
                current_time_v2 = time.time()
                silence_tracking = current_time_v2 - interruption_started - start_time

    else:
        silence_tracking = 0
        start_time = 0

    # Display the processed image
    cv2.imshow('', img_total[0][..., ::-1])
    cv2.waitKey(1)
    # Clear the image queue to process the next frame
    imgs = queue.Queue()