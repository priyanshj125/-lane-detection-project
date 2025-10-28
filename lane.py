import cv2
import numpy as np

def process_frame(frame):
    """
    Applies the full lane detection pipeline to a single video frame.
    """
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # --- 1. Grayscale and Blur ---
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise and help Canny
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # --- 2. Canny Edge Detection ---
    # Find edges in the image
    canny = cv2.Canny(blur, 50, 150)
    
    # --- 3. Region of Interest (ROI) Masking ---
    # Create a black mask with the same dimensions as the frame
    mask = np.zeros_like(canny)
    
    # Define vertices for the ROI (a trapezoid)
    # This is highly dependent on camera placement and must be tuned
    region_of_interest_vertices = [
        (0, height),                       # Bottom-left
        (int(width * 0.45), int(height * 0.6)), # Top-left
        (int(width * 0.55), int(height * 0.6)), # Top-right
        (width, height)                    # Bottom-right
    ]
    
    # Fill the polygonal mask with white
    cv2.fillPoly(mask, np.array([region_of_interest_vertices], np.int32), 255)
    
    # Apply the mask to the Canny image
    masked_edges = cv2.bitwise_and(canny, mask)
    
    # --- 4. Hough Line Transform ---
    # Detect lines in the masked edge image
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=1,             # Resolution of 'rho' in pixels
        theta=np.pi/180,   # Resolution of 'theta' in radians
        threshold=20,      # Min. number of intersections to detect a line
        minLineLength=20,  # Min. length of a line in pixels
        maxLineGap=300     # Max. allowed gap between segments
    )
    
    # Create a blank image to draw the detected lines on
    line_image = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw the line in green
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
            
    # --- 5. Combine and Overlay ---
    # Overlay the detected lines onto the original frame
    # final_frame = original * 0.8 + line_image * 1.0 + 0.0
    final_frame = cv2.addWeighted(frame, 0.8, line_image, 1.0, 0.0)
    
    return final_frame

def main():
    """
    Main function to capture video and process it.
    """
    # Try to open the default webcam (usually device 0)
    # You can replace 0 with a video file path, e.g., 'test_video.mp4'
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Opening video stream... Press 'q' to quit.")

    while cap.isOpened():
        # Read a single frame
        ret, frame = cap.read()
        
        if not ret:
            # Break the loop if we're at the end of a video file
            print("End of video stream.")
            break
        
        try:
            # Process the frame
            processed_frame = process_frame(frame)
            
            # Display the processed frame
            cv2.imshow('Lane Detection', processed_frame)
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Show the original frame if processing fails
            cv2.imshow('Lane Detection', frame)

        # Wait for 1ms and check if 'q' is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Video stream closed.")

if __name__ == "__main__":
    # To run this, you need opencv-python and numpy
    # pip install opencv-python numpy
    main()
