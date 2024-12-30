import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face and Hand modules
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the jewelry images with transparency (PNG)
ring_img = cv2.imread('ring.png', cv2.IMREAD_UNCHANGED)
necklace_img = cv2.imread('necklace.png', cv2.IMREAD_UNCHANGED)

if necklace_img is None or ring_img is None:
    print("Error: Jewelry images not loaded. Check the file paths.")
    exit()

# Adjustable scale for jewelry
necklace_scale = 0.3  # Smaller necklace size
ring_scale = 0.1  # Adjust ring size

# Function to overlay jewelry image on a specific position
def overlay_jewelry(image, x, y, jewelry_img, scale=1, alpha=1.0):
    h, w = jewelry_img.shape[:2]
    jewelry_resized = cv2.resize(jewelry_img, (int(w * scale), int(h * scale)))
    
    # Calculate placement area
    start_x = max(x - jewelry_resized.shape[1] // 2, 0)
    start_y = max(y - jewelry_resized.shape[0] // 2, 0)
    end_x = min(start_x + jewelry_resized.shape[1], image.shape[1])
    end_y = min(start_y + jewelry_resized.shape[0], image.shape[0])

    # Ensure valid region
    if end_x <= start_x or end_y <= start_y:
        return image

    # Extract ROI and apply transparency
    roi = image[start_y:end_y, start_x:end_x]
    overlay = jewelry_resized[:end_y - start_y, :end_x - start_x]
    mask = overlay[..., 3] / 255.0  # Normalize alpha channel
    inv_mask = 1.0 - mask

    # Blend images
    roi[..., :3] = (mask[..., None] * overlay[..., :3] + inv_mask[..., None] * roi[..., :3]).astype(np.uint8)

    return image

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Necklace fade-in variables
necklace_alpha = 0.0  # Start with fully transparent
fade_in_speed = 0.02  # Control how fast the necklace appears

with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh, mp_hands.Hands() as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert the image to RGB (OpenCV uses BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face and hand landmarks
        results_face = face_mesh.process(rgb_frame)
        results_hands = hands.process(rgb_frame)
        
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                # Necklace positioning
                chin_x = int(face_landmarks.landmark[152].x * frame.shape[1])
                chin_y = int(face_landmarks.landmark[152].y * frame.shape[0])
                
                # Gradually increase alpha for fade-in effect
                if necklace_alpha < 1.0:
                    necklace_alpha += fade_in_speed
                
                # Overlay necklace at the neck level with proper offset
                frame = overlay_jewelry(
                    frame, chin_x, chin_y + 70, necklace_img,
                    scale=necklace_scale, alpha=necklace_alpha
                )
        
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Detect middle finger base for better positioning
                ring_base_x = int(hand_landmarks.landmark[14].x * frame.shape[1])  # Proximal phalanx
                ring_base_y = int(hand_landmarks.landmark[14].y * frame.shape[0]) + 30
                
                # Overlay ring at the root of the middle finger
                frame = overlay_jewelry(frame, ring_base_x, ring_base_y, ring_img, scale=ring_scale)
        # Display the output
        cv2.imshow('Virtual Jewelry Try-On', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
