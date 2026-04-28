import cv2
from datetime import datetime

def take_picture(output_path=None):
    # Generate filename with timestamp if no path provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"photo_{timestamp}.jpg"
    
    # Open the default camera (0 = built-in webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access camera")
        return None
    
    # Warm up the camera (some frames are needed for auto-exposure)
    for _ in range(10):
        cap.read()
    
    # Capture the frame
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Photo saved to: {output_path}")
    else:
        print("Error: Failed to capture image")
        output_path = None
    
    # Release the camera
    cap.release()
    return output_path

if __name__ == "__main__":
    take_picture()
