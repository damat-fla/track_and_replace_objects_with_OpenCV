import cv2
import numpy as np


def detect_target_image(target_image, frame):
    
    target_shape = target_image.shape
    frame_shape = frame.shape

    # Smooth the images to reduce noise and improve keypoint detection
    # target_image = cv2.GaussianBlur(target_image, (9,9), 0).reshape(target_shape)
    # frame = cv2.GaussianBlur(frame, (9,9), 0).reshape(frame_shape)

    # KEY POINTS DETECTION USING SIFT
    sift = cv2.SIFT_create()

    # Detect key point both in frame and target image
    key_points_target = sift.detect(target_image)
    key_points_frame = sift.detect(frame)

    # Compute a descriptors for keypoints
    key_points_target, des_target = sift.compute(target_image, key_points_target)
    key_points_frame, des_frame = sift.compute(frame, key_points_frame)

    # FLANN for finding correspondences
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
    matches = flann.knnMatch(des_target, des_frame, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return key_points_target, key_points_frame, good_matches

def compute_homography(kp_target, kp_frame, matches, min_match_count=10):
    
    if len(matches) < min_match_count:
        print('not found')
        return None, None
    
    # Extract the matched keypoints' coordinates
    src_pts = np.float32([kp_target[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the homography matrix from the target image to the frame
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H, mask

def warp_image(target_image, custom_image, homography, output_shape):

    # Getting the 4 corners of the target image
    # and projecting them onto the frame using the homography
    # This way we can find the area where the custom image should be overlayed
    h, w = target_image.shape[:2]
    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]])
    dst = cv2.perspectiveTransform(np.array([pts]), homography)

    h_t, w_t = output_shape[:2]
    h, w = custom_image.shape[:2]

    # Compte the homography between the 4 corners of the custom image
    # and the area of the target image in the current frame
    pts_image = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1, 1, 2)
    homography = cv2.getPerspectiveTransform(src=pts_image, dst=dst)

    # Warp the custom image using the computed homography
    # creates a neew image which is all black except for the area where the custom image is overlayed
    warped = cv2.warpPerspective(src=custom_image, M=homography, dsize=(w_t, h_t))


    white = np.ones([h, w], dtype=np.uint8) * 255
    warp_mask = cv2.warpPerspective(white, homography, (w_t, h_t))

    return warped, warp_mask

def overlay_warped_image(frame, warped_image, warp_mask):

    # Overlay the warped image onto the frame
    warp_mask = np.equal(warp_mask, 0)
    warped_image[warp_mask] = frame[warp_mask]

    return warped_image
