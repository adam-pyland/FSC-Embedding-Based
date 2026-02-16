import cv2
import os

"""capture frames from video and save as images"""
root = '/work/Sultan/data/rotem_vids/'
dst = '/work/Sultan/data/rotem_images/'

for vid in os.listdir(root):
    if vid.endswith('.MP4'):
        vid_path = os.path.join(root, vid)
        cap = cv2.VideoCapture(vid_path)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save every 30th frame (adjust as needed)
            if frame_count % 30 == 0:
                img_name = f"{os.path.splitext(vid)[0]}_frame{frame_count}.png"
                img_path = os.path.join(dst, img_name)
                cv2.imwrite(img_path, frame)
            
            frame_count += 1
        
        cap.release()   