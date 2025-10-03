# Group-2

"FINAL CODE" folder contains the script for the experiment

Main libraries (requirement.txt included in the folder):
 - mediapipe==0.10.8
 - tensorflow-intel==2.15.0
 - threadpoolctl==3.2.0
 - google-cloud-dialogflow==2.25.0
 - google-cloud-texttospeech==2.14.2


"TEST" folder contains auixiliary scripts and discarded functionalities.

Discarded for this project but promising (and explanation):

  "Diarization with NAO.py": Performance and hardware issues. This implementation utilized ¨diart¨ library.
   - Our laptops were too slow with diarization and the delay kept stacking, made even worse when we want to stream the camera and start analyzing visual field.
   - Biggest ¨Whisper¨ model we could test on our best laptop (3050 TI GPU, 16 gb ram) was 'small'. The accuracy of the transciptions when tested on NAO was not ideal.
   - ¨Diarizatioin without NAO.py¨runs the diarization on desktop without using the NAO. 

Honorable mention to the previous versions of face detection and face tracking that we tested and refined, only to end up discarding due to compatibility issues with Dialogflow/OpenAI agents.

E.g. Retina Face ("face_detection2.py", "face_detection_nao2.py" etc.) - limited feautres for face landmarks, compatibility issues when testing with diarization (our former main functionallity).

In general, we discarded the combination of diarization + transcription + OpenAI API due to memory issues and stacking delay when adding any other functionality (e.g. visual analysis). 

Note for the future
- Ideal threshold for interruption checking: Interruption percentaje of the sentence
