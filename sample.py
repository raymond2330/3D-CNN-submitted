# Required libraries to install:
# pip install opencv-python numpy pyvirtualcam mediapipe

import cv2
import numpy as np
import pyvirtualcam
import mediapipe as mp

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Open the webcam
cap = cv2.VideoCapture(0)

# Get the camera resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Open virtual camera
with pyvirtualcam.Camera(width, height, fps=60) as cam:
    print(f'Using virtual camera: {cam.device}')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (MediaPipe uses RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform background removal
        results = selfie_segmentation.process(frame_rgb)
        mask = results.segmentation_mask

        # Smooth the mask using Gaussian blur
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Create a white background
        background = np.ones_like(frame) * 255

        # Blend the original frame with the background using the mask
        mask_3d = np.stack((mask,) * 3, axis=-1)  # Convert mask to 3 channels
        frame_no_bg = np.where(mask_3d > 0.5, frame, background)

        # Convert frame to RGB for pyvirtualcam
        frame_no_bg_rgb = cv2.cvtColor(frame_no_bg, cv2.COLOR_BGR2RGB)

        # Send frame to virtual camera
        cam.send(frame_no_bg_rgb)

        # Wait for next frame
        cam.sleep_until_next_frame()

        # Display the camera output (optional)
        cv2.imshow("Virtual Camera", frame_no_bg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
