from video_processing import load_video, extract_frames, create_video_writer, write_frame
import cv2

# This version of the code simply loads a video file, reads it frame by frame,
# and writes the unprocessed frames to a new output video using OpenCV's VideoWriter.
# No processing or modification is applied to the frames in this stage.
# It serves as a baseline for testing the video loading and saving pipeline.

in_path = './input_video.mp4'
out_path = './output_video.mp4'
codec = 'mp4v'

def main(video_path, output_path):
    video = load_video(video_path)
    
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    writer = create_video_writer(output_path, frame_size, fps, codec)

    for frame in extract_frames(video):
        # dummy: copy each frame as it is inside the writer without processing it
        write_frame(writer, frame)
        
    writer.release()


if __name__ == "__main__":
    main(in_path, out_path)