import cv2
import mediapipe as mp
import time

class FingerCounter:
    def __init__(self, max_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7, gesture_threshold=20):
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
        
        # Gesture debouncing per hand: {hand_index: (last_gesture, count)}
        self.gesture_history = {}
        self.gesture_threshold = gesture_threshold  # Number of consecutive frames to trigger an action
        
        # Map finger counts to simulated actions
        self.gesture_actions = {
            0: "Fist detected: Turning OFF lights",
            1: "Pointing detected: Activating device",
            2: "Victory sign: Scrolling UP",
            3: "Three fingers: Scrolling DOWN",
            4: "Four fingers: Adjusting volume",
            5: "Open hand detected: Turning ON lights"
        }
    
    def count_fingers(self, hand_landmarks, handedness='Right'):
        """
        Count fingers based on landmarks.
        For the thumb, the x-coordinate is compared based on handedness.
        For other fingers, compare the tip and PIP joint y-coordinates.
        """
        tip_ids = [4, 8, 12, 16, 20]   # Indexes for finger tips
        pip_ids = [2, 6, 10, 14, 18]   # Indexes for finger PIP joints
        
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

    def trigger_gesture_action(self, gesture, hand_idx):
        """
        Simulate triggering a real-life action based on the stable gesture.
        In practice, replace these print statements with calls to external systems.
        """
        action = self.gesture_actions.get(gesture, "Unknown Gesture")
        print(f"[Hand {hand_idx}] {action}")
        return action

    def process_frame(self, frame):
        # Calculate FPS for performance monitoring
        self.current_time = time.time()
        fps = 1 / (self.current_time - self.prev_time) if (self.current_time - self.prev_time) > 0 else 0
        self.prev_time = self.current_time
        
        # Convert frame from BGR to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks on the frame
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get handedness if available
                handedness_label = "Right"  # Default
                if results.multi_handedness:
                    handedness_label = results.multi_handedness[idx].classification[0].label
                
                # Count fingers for this hand
                finger_count = self.count_fingers(hand_landmarks, handedness_label)
                
                # Debounce gesture recognition per hand
                if idx not in self.gesture_history:
                    self.gesture_history[idx] = (finger_count, 1)
                else:
                    last_gesture, count = self.gesture_history[idx]
                    if last_gesture == finger_count:
                        count += 1
                    else:
                        last_gesture, count = finger_count, 1
                    self.gesture_history[idx] = (last_gesture, count)
                    
                    # Trigger action if gesture is stable over threshold frames
                    if count == self.gesture_threshold:
                        action_text = self.trigger_gesture_action(finger_count, idx)
                        # Reset counter to avoid multiple triggers
                        self.gesture_history[idx] = (finger_count, 0)
                        # Display the action on the frame
                        cv2.putText(frame, action_text, 
                                    (10, 150 + idx * 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                # Display finger count and handedness on the frame
                cv2.putText(frame, f"{handedness_label} Hand: {finger_count}", 
                            (10, 50 + idx * 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        # Overlay FPS on the frame
        cv2.putText(frame, f"FPS: {int(fps)}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
        return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    finger_counter = FingerCounter(max_hands=2)
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Unable to read frame from camera.")
            break
            
        # Optionally, you can resize the frame for performance:
        # frame = cv2.resize(frame, (640, 480))
            
        frame = finger_counter.process_frame(frame)
        cv2.imshow("Real-Life Gesture Control", frame)
        
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