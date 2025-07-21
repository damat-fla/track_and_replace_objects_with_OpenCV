import cv2
import numpy as np

# Given a path, returns the VideoCapture object
def load_video(path: str) -> cv2.VideoCapture:
    
    cap = cv2.VideoCapture(filename=path)

    # Check if camera opened successfully
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    return cap

# Given a VideoCapture object, generate frames one by one
def extract_frames(video: cv2.VideoCapture, rotate_code: int = None):
    
    while True:
        # ret: a boolean value stating if the frame is correctly read
        # frame: an image (np.ndarray)
        ret, frame = video.read()

        if not ret: # check if we are at the end of video or something gone wrong
            break 

        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        yield frame

    video.release()

# Create a cv2.VideoWriter object to store processed frames
def create_video_writer(output_path: str, frame_size: tuple[int, int], fps: int = 30, codec: str = 'mp4v') -> cv2.VideoWriter:
    
    # Specify the Codec (COder-DECoder) algorithm of the video 
    # * explodes the sting in 4 different characters since VideoWriter_fourcc expects 4 parameters
    fourcc = cv2.VideoWriter_fourcc(*codec)

    # Creates the VideoWriter object
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size, True)
    
    if not writer.isOpened():
        raise IOError(f"Cannot create video writer for: {output_path}")
    
    return writer

# Append a new frame to the VideoWriter
def write_frame(writer: cv2.VideoWriter, frame: np.ndarray):
    writer.write(frame)


