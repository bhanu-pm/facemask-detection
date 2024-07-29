import cv2
import mediapipe as mp

import draw
import predictor

mpFaceDetection = mp.solutions.face_detection
faceDetect = mpFaceDetection.FaceDetection(0.65)

def cam_detector():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        isTrue, frame = cap.read()
        results = faceDetect.process(frame)

        # Face Detection
        if results.detections:
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                xcor = int(bboxC.xmin * iw)
                ycor = int(bboxC.ymin * ih) - 30
                width = int(bboxC.width * iw)
                height = int(bboxC.height * ih) + 30
                bbox = [xcor, ycor, width, height]

                # Predicting the result of the Face on Screen
                result = predictor.predict_on_face(frame, xcor, ycor, width, height)

                # Drawing the rectangle on detected face along with result
                draw.drawing_on_frame(frame, xcor, ycor, width, height, result)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
