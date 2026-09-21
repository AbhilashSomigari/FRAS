# attendance_system.py - Handles face recognition and attendance marking

import cv2
import csv
import numpy as np
import face_recognition
import os
import time
from datetime import datetime

class AttendanceSystem:
    def __init__(self):
        self.dataset_path = 'dataset'
        self.attendance_log = 'attendance_log.csv'
        self.cooldown_period = 75 * 60  # 75 minutes in seconds
        self.student_names = []
        self.face_encodings = []
        self.attendance_records = {}
        self.initialize_system()

    def initialize_system(self):
        """Initialize the system by creating necessary files"""
        # Create attendance log if it doesn't exist
        if not os.path.exists(self.attendance_log):
            with open(self.attendance_log, 'w') as f:
                f.write('Name,Date,Time\n')
            print(f"Created attendance log file: {self.attendance_log}")
        else:
            self.load_attendance_records()

    def load_attendance_records(self):
        """Seed the in-memory cooldown tracker from the existing log, so a restart
        mid-session doesn't allow a student to be marked twice within the cooldown window"""
        with open(self.attendance_log, 'r') as f:
            for row in csv.DictReader(f):
                marked_at = datetime.strptime(f"{row['Date']} {row['Time']}", "%Y-%m-%d %H:%M:%S").timestamp()
                if marked_at > self.attendance_records.get(row['Name'], 0):
                    self.attendance_records[row['Name']] = marked_at
    
    def load_dataset(self):
        """Load student images from dataset folder and encode faces.

        Every captured angle is encoded and added as its own known-face entry
        (rather than just the first image), so recognition can match a student
        from any of the poses/lighting conditions captured at registration.
        """
        print("Loading dataset...")
        self.student_names = []
        self.face_encodings = []

        student_folders = [f for f in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, f))]

        if not student_folders:
            print("No student folders found in dataset. Please register students first.")
            return False

        for student_folder in student_folders:
            folder_path = os.path.join(self.dataset_path, student_folder)
            student_name = student_folder  # Assuming folder name is student name

            # Get all images for each student
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                print(f"No images found for student {student_name}")
                continue

            encodings_added = 0
            for image_file in image_files:
                img_path = os.path.join(folder_path, image_file)
                img = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(img)

                if not face_locations:
                    print(f"No face detected in {img_path}")
                    continue

                encoding = face_recognition.face_encodings(img, face_locations)[0]

                self.student_names.append(student_name)
                self.face_encodings.append(encoding)
                encodings_added += 1

            if encodings_added == 0:
                print(f"No usable face images for student {student_name}")

        unique_students = len(set(self.student_names))
        print(f"Dataset loaded successfully with {unique_students} students ({len(self.face_encodings)} face encodings)")
        return len(self.face_encodings) > 0
    
    def mark_attendance(self, name):
        """Mark attendance for a student"""
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        time_string = now.strftime("%H:%M:%S")
        
        current_time = time.time()
        
        # Check if student has already been marked within cooldown period
        if name in self.attendance_records:
            last_marked = self.attendance_records[name]
            if current_time - last_marked < self.cooldown_period:
                # Student already marked within cooldown period
                return False
                
        # Mark attendance
        with open(self.attendance_log, 'a') as f:
            f.write(f"{name},{date_string},{time_string}\n")
            
        # Update attendance record with current timestamp
        self.attendance_records[name] = current_time
        
        print(f"Attendance marked for {name} at {time_string}")
        return True
    
    def run_attendance_system(self):
        """Run the attendance system using webcam"""
        if not self.load_dataset():
            print("Cannot start attendance system. No students registered.")
            return
            
        print("Starting attendance system...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
            
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                    # Scale back face locations
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Compare with known faces
                    matches = face_recognition.compare_faces(self.face_encodings, face_encoding)
                    name = "Unknown"
                    attendance_marked = False
                    
                    if True in matches:
                        # Find best match
                        face_distances = face_recognition.face_distance(self.face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = self.student_names[best_match_index]
                            attendance_marked = self.mark_attendance(name)
                    
                    # Set rectangle color based on whether attendance was marked
                    rect_color = (0, 255, 0) if (name != "Unknown" and attendance_marked) else (0, 0, 255)
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (left, top), (right, bottom), rect_color, 2)
                    
                    # Display name and status
                    status = "Marked" if (name != "Unknown" and attendance_marked) else "Already Marked" if name != "Unknown" else "Unknown"
                    cv2.putText(frame, f"{name} - {status}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2)
            
            # Display the frame
            cv2.imshow("Attendance System", frame)
            
            # Capture key press and check for 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Attendance system closed.")
                break
        
        cap.release()
        cv2.destroyAllWindows()