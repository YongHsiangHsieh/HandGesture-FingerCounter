import cv2
import mediapipe as mp
import time

class FingerCounter:
    def __init__(self, max_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        # Initialize Mediapipe Hands module with configurable parameters
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        # Variables for calculating FPS
        self.prev_time = time.time()
        self.current_time = time.time()
    
    def count_fingers(self, hand_landmarks, handedness='Right'):
        """
        Count fingers based on landmarks.
        For the thumb, the x-coordinate is compared based on handedness.
        For other fingers, compare the tip and pip joint y-coordinates.
        """
        tip_ids = [4, 8, 12, 16, 20]  # Indexes for finger tips
        pip_ids = [2, 6, 10, 14, 18]  # Indexes for finger PIP joints
        
        count = 0
        
        # Check thumb separately; adjust logic for right/left hand
        if handedness == 'Right':
            if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[pip_ids[0]].x:
                count += 1
        else:  # Left hand
            if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[pip_ids[0]].x:
                count += 1
        
        # Check other fingers
        for i in range(1, 5):
            if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[pip_ids[i]].y:
                count += 1
                
        return count

    def process_frame(self, frame):
        # Calculate and display FPS for performance monitoring
        self.current_time = time.time()
        fps = 1 / (self.current_time - self.prev_time) if (self.current_time - self.prev_time) > 0 else 0
        self.prev_time = self.current_time
        
        # Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks and connections on the frame
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Use Mediapipe’s handedness data (if available)
                handedness_label = "Right"  # Default assumption
                if results.multi_handedness:
                    handedness_dict = results.multi_handedness[idx].classification[0]
                    handedness_label = handedness_dict.label
                
                # Count fingers for this hand
                finger_count = self.count_fingers(hand_landmarks, handedness_label)
                
                # Display finger count and handedness on the frame
                cv2.putText(frame, f"{handedness_label} Hand: {finger_count}", 
                            (10, 50 + idx * 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Example gesture recognition: display messages for specific gestures
                if finger_count == 0:
                    cv2.putText(frame, "Fist Detected", 
                                (10, 100 + idx * 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif finger_count == 5:
                    cv2.putText(frame, "Open Hand Detected", 
                                (10, 100 + idx * 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
        # Overlay the FPS on the frame
        cv2.putText(frame, f"FPS: {int(fps)}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
        return frame

def main():
    cap = cv2.VideoCapture(0)
    finger_counter = FingerCounter(max_hands=2)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Process each frame to detect hands and count fingers
        frame = finger_counter.process_frame(frame)
        cv2.imshow("Enhanced Finger Counter", frame)
        
        # Check key presses: 'q' to quit, 's' to save a screenshot
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("screenshot.png", frame)
            print("Screenshot saved!")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()