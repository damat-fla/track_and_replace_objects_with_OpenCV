# track_and_replace_objects_with_OpenCV
This repository contains a classical computer vision pipeline (*no machine learning or deep learning involved*) that detects and tracks a planar target object in videos and replaces it with a custom image. It leverages feature detection, descriptor matching, and homography estimation for each frame to achieve augmented reality effects.

<div align="center">
  <h3>Input vs Output</h3>
  <img src="./data/gifs/input_output.gif" width="400" alt="Input and Output">
</div>

---

### Usage

1. Install the required libraries listed in `requirements.txt` (if necessary):
      ```bash
      pip install -r requirements.txt
      ```

2. Place your input video file, target image, and custom overlay image inside the `./data/` folder. Rename them `input_video`, `target` and `custom` respectively. The allowed image extensions are `jpg`, `jpeg` and `png`. Only `mp4` format is allowed for videos.
3. Run the main script: `python main.py`

---

### Approach

The general pipeline is as follows:

1. **Key Points detection and description**:  In each frame, the system detects the presence of a target planar object using SIFT (Scale-Invariant Feature Transform) keypoints and descriptors.

2. **Feature Matching**:  FLANN-based matcher identifies correspondences between the target image and the current frame.

3. **Homography Estimation**:  Using RANSAC, a homography matrix is computed to map the target image onto the frame.

4. **Image Warping**:  A custom image is warped to fit the perspective of the detected object.

5. **Overlay**:  The warped custom image is blended into the original frame, replacing the target object.

6. **Homography Smoothing**:  Temporal smoothing is applied between frames to reduce visual jitter.

---

### Project Structure

- `video_processing.py`:  Contains utility functions to:
  - Load video, extract frames and return them one by one
  - Initialize the video writer
  - Write processed frames to output

- `frame_processing.py`:  Contains the core computer vision logic:
  - Detect keypoints and compute descriptors
  - Match features and compute homography
  - Warp the custom image and overlay it
  - Apply homography smoothing

- `utils.py`: Contains helper functions for handling file paths and formats

- `main.py`: The main entry point of the pipeline. It orchestrates the entire process by loading the input data, calling the frame processing functions, and writing the final output video.

