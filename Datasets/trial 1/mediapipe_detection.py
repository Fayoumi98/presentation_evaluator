import mediapipe as mp  
import cv2 


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# capture video from webcame at adress 0
# if there are multiple cameras adress 0,1,2,3,...
cap = cv2.VideoCapture(0)



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
                        
        cv2.imshow('Output Plot', image)
        
        # if q key is pressed close camera feed opencv image window
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
