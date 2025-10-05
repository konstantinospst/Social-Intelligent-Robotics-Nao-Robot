import mediapipe as mp
import cv2

def detector(image):
    # Face Detection initialization
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection, \
        mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True) as face_mesh:

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detection.process(image_rgb)
        user_distances = {}
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Process face mesh
                results2 = face_mesh.process(image_rgb) 
                if results2.multi_face_landmarks:
                    for face_id, face_landmarks in enumerate(results2.multi_face_landmarks):
                        pt1 = face_landmarks.landmark[0]
                        x1 = int(pt1.x * iw)
                        y1 = int(pt1.y * ih)
                        pt2 = face_landmarks.landmark[17]
                        x2 = int(pt2.x * iw)
                        y2 = int(pt2.y * ih)
                        # Calculate lip distance for each face
                        lip_distance = abs(y2 - y1)
                        user_distances[face_id] = lip_distance
                        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return image, user_distances