import cv2
import numpy as np
from retinaface import RetinaFace
import queue
import threading
from sic_framework.core.message_python2 import CompressedImageMessage
from sic_framework.devices import Nao

# Initialize RetinaFace
detector = RetinaFace(quality="low")

# Queue for storing images
imgs = queue.Queue()

# Function for face detection and drawing rectangles
def detect_faces(image):
    # Perform face detection using RetinaFace
    faces = detector.predict(image)

    # Draw rectangles around detected faces and red dots at their centers
    for face in faces:
        x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']

        # Calculate width and height from x1, y1, x2, y2
        w = x2 - x1
        h = y2 - y1

        # Calculate the center of the rectangle
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Draw rectangle using the converted coordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Draw a red dot at the center of the rectangle
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)  # -1 indicates a filled circle

    return image

# Callback function for receiving images from Nao's camera
def on_image(image_message: CompressedImageMessage):
    # Get the image from the message
    img = image_message.image
    
    # # Convert BGR to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Put the image into the queue for processing
    imgs.put(img)
    
# Your existing code for Nao and setting up the callback
nao = Nao(ip="192.168.0.151")
nao.top_camera.register_callback(on_image)

while True:
        img = imgs.get()    
            
        # Perform face detection
        image_with_faces = detect_faces(img)
        
        cv2.imshow('', image_with_faces[..., ::-1])  # cv2 is BGR instead of RGB
        # cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
