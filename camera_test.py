import cv2
import time

def test_camera():
    print("Camera Test Starting...")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test multiple camera indices
    for camera_id in [0, 1, -1]:
        print(f"\nTesting camera {camera_id}...")
        
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            print(f"SUCCESS: Camera {camera_id} opened!")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"SUCCESS: Frame captured! Shape: {frame.shape}")
                
                # Create window and show frame
                window_name = f"Camera {camera_id} Test"
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(window_name, frame)
                
                print(f"Window '{window_name}' should be visible now!")
                print("Press 's' to save frame, 'q' to next camera, ESC to exit...")
                
                # Wait for key press
                frame_counter = 0
                while True:
                    key = cv2.waitKey(30) & 0xFF
                    
                    # Read new frame
                    ret, frame = cap.read()
                    if ret:
                        frame_counter += 1
                        # Add frame counter to image
                        cv2.putText(frame, f"Frame: {frame_counter}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow(window_name, frame)
                    
                    if key == ord('q'):
                        print("Moving to next camera...")
                        break
                    elif key == ord('s'):
                        filename = f"test_frame_camera_{camera_id}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"Frame saved as {filename}")
                    elif key == 27:  # ESC key
                        print("Test stopped by user")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                
                cv2.destroyWindow(window_name)
            else:
                print(f"ERROR: Camera {camera_id} opened but cannot read frames")
        else:
            print(f"ERROR: Camera {camera_id} failed to open")
        
        cap.release()
    
    cv2.destroyAllWindows()
    print("\nCamera test complete!")

if __name__ == "__main__":
    test_camera()