import cv2
import os 
import pdb

def create_video(images_folder, output_video_path, fps=30):
    images = [img for img in os.listdir(images_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    frame = cv2.imread(os.path.join(images_folder, images[0]))
    height, width, _ = frame.shape

    # video = cv2.VideoWriter(output_video_path, 0, fps, (width, height))
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(images_folder, image)))

    cv2.destroyAllWindows()
    video.release()