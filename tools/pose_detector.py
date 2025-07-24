import sys
import base64
import json
import cv2
import numpy as np
import mediapipe as mp

def main():
    try:
        input_data = sys.stdin.read()
        parsed = json.loads(input_data)
        base64_image = parsed['image']

        # Decode base64 to image
        image_bytes = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Setup MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True)
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Check and print landmarks
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                print(f"Landmark {idx + 1}: x={x}, y={y}")
        else:
            print("No pose landmarks detected.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
