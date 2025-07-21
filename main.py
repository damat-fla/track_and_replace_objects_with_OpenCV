from video_processing import load_video, extract_frames, create_video_writer, write_frame
from frame_processing import compute_homography, detect_target_image, warp_image, overlay_warped_image, smooth_homography
from tqdm import tqdm
import cv2


# This version of the code implement the entire pipeline to process 
# a video by overlaying a custom image onto a target image. To do so,
# it uses OpenCV to read the video, extract frames, detect keypoints
# both in the target image and the current frame, compute the homography.
# To avoid sudden jumps, it smooths the homography between the current
# and the previous homography. 
# Then it projects the 4 corners of the target image onto the current frame
# to detect the area where the custom image should be overlayed.
# After that, it computes a new homography between the 4 corners of the 
# custom image and the area of the target image in the current frame.
# Finally, it warps the custom image and overlays it onto the current frame.
# This is a very basic implementation and requires many adjustments and improvements.

in_path = './data/input_video.mp4'
out_path = './data/output_video.mp4'
target_image_path = './data/model.jpg'
custom_image_path = './data/chill_guy.png'
codec = 'mp4v'

target = cv2.imread(target_image_path)
custom = cv2.imread(custom_image_path)

def main(video_path, output_path):

    H_prev = None

    print(f"Opening video: {video_path}")

    video = load_video(video_path)
    
    # Get video properties
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    writer = create_video_writer(output_path, frame_size, fps, codec)
    
    print("Processing frames...")

    # Loop through each frame in the video using tqdm for progress bar
    for frame in tqdm(extract_frames(video, rotate_code=cv2.ROTATE_90_CLOCKWISE), desc="Progress", total=total_frames):

        key_points_target, key_points_frame, good_matches = detect_target_image(target, frame)
        H, mask = compute_homography(key_points_target, key_points_frame, good_matches)
       
        # Smooth the homography to avoid sudden jumps: gives a more stable overlay
        H_smooth = smooth_homography(H, H_prev, alpha=0.6)  # Smoothing
        H_prev = H_smooth

        warped, warp_mask = warp_image(target, custom, H_prev, frame.shape)
        warped = overlay_warped_image(frame, warped, warp_mask)
        
        write_frame(writer, warped)
        
    writer.release()

    print(f"Video processing completed. Output saved to {output_path}")


if __name__ == "__main__":
    main(in_path, out_path)