import pickle
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np



with open('model1.pkl', 'rb') as f:
    model = pickle.load(f)
    


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic



# capture video from webcame at adress 0
# if there are multiple cameras adress 0,1,2,3,...
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)



# Initiate holistic model
# the minimum prediction percentage of the holistic is 50% as for the minimum tracking prediction percentage is 50%
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # while the feed is on get the camera frame
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor the camera frame from BGR to RGB
        # Note Mediapipe take frame with color channel RGB and OpenCV display the image as BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        # Make Detections using the holistic model 
        results = holistic.process(image)
        
        #print(results.face_landmarks)
        # Results Contains:
        # person detected:
        
        # A) "pose_landmarks" field that contains the pose landmarks.
        # B) "pose_world_landmarks" field that contains the pose landmarks in real-world 3D coordinates that are in meters with the origin at the center between hips.
        # C) "left_hand_landmarks" field that contains the left-hand landmarks.
        # D) "right_hand_landmarks" field that contains the right-hand landmarks.
        # E) "face_landmarks" field that contains the face landmarks.
        # F) "segmentation_mask" field that contains the segmentation mask if "enable_segmentation" is set to true.
        
        
        # Recolor image back from RGB to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # To draw any landmark you will need 
        # A- The frame to draw on
        # B- The Landmarks to draw
        # C- The connections between those landmarks
        # D- landmark Drawing Specs (color RGB / line thickness / Circle Radius)
        # E- Links/Connections Drawing Specs (color RGB / line thickness / Circle Radius)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections (Body)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
                        

        # Export Landmarks Coordinates 
        try:
            
            
            
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
            # Concate rows to get all the data in one row this can be splited in the future to (pose/face/right_hand/left_hand)
            row = pose_row + face_row
            

            
            # Make Detections
            
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)
            
            
            
            # Grab ear coords
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))
            

            cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
            
            cv2.putText(image, body_language_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            
            
        except:
            pass
        
        
        cv2.imshow('Display Reults', image)


        # if q key is pressed close camera feed opencv image window
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        

        
cap.release()
cv2.destroyAllWindows()
