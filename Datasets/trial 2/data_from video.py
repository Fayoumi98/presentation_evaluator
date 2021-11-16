import mediapipe as mp  
import cv2 
import csv
import os
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


cap = cv2.VideoCapture("4.mp4")

# classes of the data
class_name = "back"



while cap.isOpened():
    
    # Initiate holistic model
    # the minimum prediction percentage of the holistic is 50% as for the minimum tracking prediction percentage is 50%
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
        ret, frame = cap.read()
        
        if ret == False:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = holistic.process(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
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
                        
        cv2.imshow('Output Plot', image)
        #print(results)
            
        # Export Landmarks Coordinates 
        try:
            
            
            
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            print(pose_row)
            
            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
            
            
            # Concate rows to get all the data in one row this can be splited in the future to (pose/face/right_hand/left_hand)
            row = pose_row + face_row
            
            
            # Append class name in the first column every row
            row.insert(0, class_name)
            
            # Export to CSV
            with open('data.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 
                        
        except:
            pass
    
    
    
    
    
        # if q key is pressed close camera feed opencv image window
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    
            
cap.release()
cv2.destroyAllWindows()
