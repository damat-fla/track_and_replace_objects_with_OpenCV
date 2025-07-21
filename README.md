# track_and_replace_objects_with_OpenCV
This repository contains a classical computer vision pipeline (*no machine learning or deep learning involved*) that detects and tracks a planar target object in videos and replaces it with a custom image. It leverages feature detection, descriptor matching, and homography estimation for each frame to achieve augmented reality effects.

<div align="center">
  <h3>Input vs Output</h3>
  <img src="./data/gifs/input_output.gif" width="400" alt="Input and Output">
</div>

### Usage

1. Install the required libraries listed in `requirements.txt` (if necessary):
      ```bash
      pip install -r requirements.txt
      ```
2. Place your input video file, target image, and custom overlay image inside the `./data/` folder. Rename them `input_video`, `target` and `custom` respectively. The allowed image extensions are `jpg`, `jpeg` and `png`. Only `mp4` format is allowed for videos.
3. Run the main script: `python main.py`
