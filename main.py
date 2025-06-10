from video_processing import load_video, extract_frames, create_video_writer, write_frame
from frame_processing import compute_homography, detect_target_image, warp_image, overlay_warped_image
import cv2

# This version of the code implement the entire pipeline to process 
# a video by overlaying a custom image onto a target image. To do so,
# it uses OpenCV to read the video, extract frames, detect keypoints
# both in the target image and the current frame, compute the homography.
# Then it projeects the 4 corners of the target image onto the current frame
# to detect the area where the custom image should be overlayed.
# After thet, it comptes a new homography between the 4 corners of the 
# custom image and the area of the target image in the current frame.
# Finally, it warps the custom image and overlays it onto the current frame.
# This is a very basic implementation and requires many adjustments and improvements.

in_path = './input_video.mp4'
out_path = './output_video.mp4'
target_image_path = './model.jpg'
custom_image_path = './chill_guy.png'
codec = 'mp4v'

target = cv2.imread(target_image_path)
custom = cv2.imread(custom_image_path)

def main(video_path, output_path):
    video = load_video(video_path)
    
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    writer = create_video_writer(output_path, frame_size, fps, codec)

    count = 0
    for frame in extract_frames(video):
        
        print(f"Processing frame {count}")
        count += 1
        key_points_target, key_points_frame, good_matches = detect_target_image(target, frame)
        H, mask = compute_homography(key_points_target, key_points_frame, good_matches)
        warped, warp_mask = warp_image(target, custom, H, frame.shape)
        warped = overlay_warped_image(frame, warped, warp_mask)
        
        write_frame(writer, warped)
        
    writer.release()


if __name__ == "__main__":
    main(in_path, out_path)