import cv2
import numpy as np
from fer import FER
import matplotlib.pyplot as plt

def analyze_webcam_emotions():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize the emotion detector
    emotion_detector = FER(mtcnn=True)
    
    print("Press 'q' to quit")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
            
        # Detect emotions in the frame
        result = emotion_detector.detect_emotions(frame)
        
        # If a face is detected
        if result:
            # Get the first face detected
            face = result[0]
            emotions = face['emotions']
            
            # Get the bounding box
            x, y, w, h = face['box']
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Find the dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            confidence = emotions[dominant_emotion]
            
            # Display emotion text
            text = f"{dominant_emotion}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)
            
            # Print emotions to console
            print("\nDetected emotions:")
            for emotion, score in emotions.items():
                print(f"{emotion}: {score:.2f}")
        
        # Display the frame
        cv2.imshow('Emotion Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_webcam_emotions()
