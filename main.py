import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import face_recognition
import os
from datetime import datetime
import requests

# Path to the directory containing reference images
reference_images_dir = "Students/"

# Initialize MTCNN for face detection
detector = MTCNN()

# Load all reference images, compute embeddings, and store them
reference_embeddings = []
reference_names = []
for file_name in os.listdir(reference_images_dir):
    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(reference_images_dir, file_name)
        image = face_recognition.load_image_file(image_path)
        
        # Detect the face and get the embedding
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            reference_embeddings.append(face_encoding)
            reference_names.append(os.path.splitext(file_name)[0])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

frame_count = 0
faces = []  # Initialize faces variable

def is_within_active_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    current_day = now.weekday()  # Monday is 0 and Sunday is 6
    return (current_time >= "07:30" and current_time <= "23:59" and current_day in range(5))  # Monday to Friday

while True:
    if is_within_active_time():
        ret, frame = cap.read()
        frame_count += 1

        if ret:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (640, 480))

            # Detect faces every 5th frame
            if frame_count % 5 == 0:
                faces = detector.detect_faces(small_frame)
            
            for face in faces:
                x, y, w, h = face['box']
                
                # Draw a rectangle around the face
                cv2.rectangle(small_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract the face ROI
                face_roi = small_frame[y:y+h, x:x+w]
                
                # Convert ROI to RGB
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                
                # Detect the face location and compute the embedding
                face_locations = face_recognition.face_locations(face_rgb)
                if face_locations:
                    face_encoding = face_recognition.face_encodings(face_rgb, face_locations)[0]
                    
                    # Compare the face embedding with reference embeddings
                    matches = face_recognition.compare_faces(reference_embeddings, face_encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(reference_embeddings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = reference_names[best_match_index]
                        
                        # Retrieve the current date and time
                        current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
                        print(f"Sending date_time_in: {current_time}")
                        is_attend = 1
                        checkpoint_id = 2
                        attendance_timetable_id = 1
                        date_time_out = ""
                        platform = "FR"
                        
                        # Initialize attendance_status
                        attendance_status = "Processing..."

                        # Send the name to the API
                        findStudentURL = "http://192.168.0.123:8000/findStudents"
                        data = {"name": name}
                        try:
                            response = requests.post(findStudentURL, json=data)
                            if response.status_code == 200:
                                # Parse the response to get student_study_session_id
                                response_json = response.json()
                                if response_json:
                                    student_study_session_id = response_json[0]["student_study_session_id"]
                                    print(f"Student Study Session ID: {student_study_session_id}")
                                    
                                    recordAttendanceURL = "http://192.168.0.123:8000/recordAttendance"
                                    
                                    sendData = {
                                        "date_time_in": current_time,
                                        "is_attend": is_attend,
                                        "checkpoint_id": checkpoint_id,
                                        "attendance_timetable_id": attendance_timetable_id,
                                        "student_study_session_id": student_study_session_id,
                                        "platform": platform
                                    }
                                    
                                    try:
                                        response = requests.post(recordAttendanceURL, json=sendData)
                                        print(f"Data Send: {sendData}")
                                        if response.status_code == 200:
                                            attendance_status = "Successfully Recorded"
                                        elif response.status_code == 404:
                                            attendance_status = "Attendance Already Recorded"
                                        elif response.status_code == 407:
                                            attendance_status = "Waiting For Another Method"
                                        else:
                                            attendance_status = "Attendance Can't Be Recorded"  
                                    except requests.exceptions.RequestException as e:
                                        print(f"Error connecting to API: {e}")
                                        attendance_status = "Connection Error"
                                else:
                                    print("No data found in API response.")
                                    attendance_status = "No Data Found"
                            else:
                                print(f"Failed to send data to API: {response.status_code} {response.text}")
                                attendance_status = "API Error"
                        except requests.exceptions.RequestException as e:
                            print(f"Error connecting to API: {e}")
                            attendance_status = "Connection Error"
                        
                        # Display the name above the rectangle
                        cv2.putText(small_frame, name.upper(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # Display the attendance status below the name
                        cv2.putText(small_frame, attendance_status, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(small_frame, "UNKNOWN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # Display the result
            cv2.imshow('Face Recognition', small_frame)
    else:
        # If not within active time, show a message or a blank frame
        blank_frame = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(blank_frame, "System Inactive", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Face Recognition', blank_frame)
        
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
