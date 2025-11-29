import cv2
from PIL import Image
import pathlib


def video_to_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
        img.save(f"{output_folder}/frame_{frame_count:05d}.png")
        frame_count += 1

    cap.release()


if __name__ == "__main__":
    video_path = "data/0/16.avi"
    output_folder = "data/0/frames"
    video_to_frames(video_path, output_folder)
