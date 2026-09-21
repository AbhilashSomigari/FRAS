# student_registration.py - Handles student registration and image capture

import cv2
import os
import face_recognition
import time

class StudentRegistration:
    def __init__(self):
        self.dataset_path = 'dataset'
        self.initialize_directory()
    
    def initialize_directory(self):
        """Create dataset directory if it doesn't exist"""
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            print(f"Created dataset directory at {self.dataset_path}")
    
    def register_student(self, spacebar_capture=True, max_images=10):
        """
        Register a new student by capturing face images
        
        Args:
            spacebar_capture (bool): If True, capture image on spacebar press
            max_images (int): Maximum number of images to capture
        """
        student_name = input("Enter student name: ")

        # Create student directory if it doesn't exist
        student_dir = os.path.join(self.dataset_path, student_name)
        if not os.path.exists(student_dir):
            os.makedirs(student_dir)

        # Offset new filenames past any images already captured for this student,
        # so re-registering the same name doesn't overwrite earlier captures
        existing_images = len([f for f in os.listdir(student_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

        print(f"Starting registration for {student_name}...")
        
        if spacebar_capture:
            print(f"Press SPACEBAR to capture an image (max {max_images} images)")
            print("Press 'q' to quit")
        else:
            print(f"We will capture {max_images} images automatically. Please look at the camera.")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return False
            
        count = 0
        
        while count < max_images:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Detect faces in the frame (face_recognition expects RGB, OpenCV frames are BGR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Display the frame with face rectangle
            display_frame = frame.copy()
            if face_locations:
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add instruction text
            instruction = f"Press SPACE to capture ({count}/{max_images})" if spacebar_capture else f"Capturing: {count}/{max_images}"
            cv2.putText(display_frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
            
            # Display the frame
            cv2.imshow("Registration", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Check for 'q' key first - exit immediately when pressed
            if key == ord('q'):
                print("Registration cancelled by user.")
                break
                
            # Capture image on spacebar press or automatically
            if (spacebar_capture and key == 32 and face_locations) or (not spacebar_capture and face_locations):
                # Save the image
                img_name = os.path.join(student_dir, f"{student_name}_angle{existing_images+count+1}.jpg")
                cv2.imwrite(img_name, frame)
                print(f"Image {count+1}/{max_images} saved: {img_name}")
                count += 1
                
                # Add a short delay if automatic capture
                if not spacebar_capture:
                    time.sleep(0.5)
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Registration completed for {student_name}. {count} images captured.")
        return True