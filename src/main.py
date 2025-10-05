import queue
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import threading

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

#### Dialogflow calling function(s)
def on_dialog(message):
    if message.response:
        if message.response.recognition_result.is_final:
            print("Transcript:", message.response.recognition_result.transcript)

keyfile_json = json.load(open("sictest-404013-2a1d5afc360b.json"))

# set up the config
conf = DialogflowConf(keyfile_json=keyfile_json, sample_rate_hertz=44100, language="en")

# initiate Dialogflow object
dialogflow = Dialogflow(ip='localhost', conf=conf)
            
def on_image(image_message: CompressedImageMessage):
    # we could use cv2.imshow here, but that does not work on Mac OSX
    imgs.put(image_message.image)    

dialogflow_request_queue = queue.Queue()

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
                 
nao = Nao(ip="10.0.0.121")

# local desktop setup
desktop = Desktop()

#### Dialogflow chunk ####
dialogflow.connect(desktop.mic)

# register a callback function to act upon arrival of recognition_result
dialogflow.register_callback(on_dialog)

x = np.random.randint(10000)

dialogflow_thread = threading.Thread(target=dialogflow_request_handler)
dialogflow_thread.start()

nao.motion.request(NaoPostureRequest("Stand", 1))
# Here the leds turn blue
reply = nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 0, 1, 0))
nao.motion.request(NaoqiBreathingRequest("Body", True))
nao.autonomous.request(NaoBasicAwarenessRequest(True))

# Threshold for lip distance fluctuations for each user
user_threshold = 10

imgs=queue.Queue()

nao.top_camera.register_callback(on_image)

start_time = 0
silence_tracking = 0
interruption_started=0
silence_threshhold = 0.3
while True:
    # Here the leds turn blue
    reply = nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 0, 1, 0))
    
    img = imgs.get()
    img = cv2.flip(img,0)
    img_total = detector(img)
    user_distances = img_total[1]
    
    user_talking = []
    for user_id, lip_distance in user_distances.items():
        if lip_distance > user_threshold:
            user_talking.append("true")
            
    if (len(user_talking)== 1 and silence_tracking <=silence_threshhold)   or (len(user_talking)==2):
        dialogflow_request_queue.put(x)
        if len(user_talking) >= 2:
            silence_tracking = 0
            # Update start_time only when both users are talking
            if start_time == 0:
                # Record start time when condition is met
                start_time = time.time()    
            else:
                current_time = time.time()
                elapsed_time = current_time - start_time + silence_tracking
                if elapsed_time >= 3:
                    print("There is an interruption")
                    # Reset start_time after interruption
                    start_time = 0
                    
                    # Here the leds turn red
                    reply = nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 0, 0, 0))
                    nao.tts.request(NaoqiTextToSpeechRequest("I think there may have been an interruption. When the eyes turn blue, please try again?", animated=True))
                    time.sleep(3)
        elif silence_tracking>silence_threshhold:
            start_time=0
            silence_tracking=0
        else:
            if  interruption_started==0:
                interruption_started = time.time()
            else:
                current_time_v2 = time.time()
                silence_tracking = current_time_v2 - interruption_started - start_time
                
    else:
        silence_tracking=0
        start_time=0        

    cv2.imshow('',img_total[0][...,::-1])
    cv2.waitKey(1)
    imgs=queue.Queue()