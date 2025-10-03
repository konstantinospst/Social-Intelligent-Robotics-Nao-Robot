from sic_framework.devices import Nao
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.services.dialogflow.dialogflow import (DialogflowConf, GetIntentRequest, RecognitionResult,
                                                          QueryResult, Dialogflow)
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoBasicAwarenessRequest

nao = Nao(ip="192.168.0.148")

# Requesting NaoBasicAwarenessRequest
reply = nao.autonomous.request(NaoBasicAwarenessRequest(True, "People", "FullyEngaged", "Head"))