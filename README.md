<br />
<p align="center">
  <img src="https://github.com/vocdex/vocdex.github.io/blob/master/assets/img/icon.png" width="150" title="hover text">
</p>

# DIGIT
This codebase allows you:
- Collect image frames from DIGIT and annotate circles in each frame.
- Save the annotated frame values into a csv file.
- Train a baseline MLP model for RGB to Normal mapping.
- Generate depth maps in real-time using a fast Poisson Solver.
- Estimate 2D object pose using PCA and OpenCV built-in algorithms.

Currently, labeling circles is done manually for each sensor. It can take up to an hour for annotating 30 images.  
This codebase has a script that will replace manual labeling and model training process up to 15 mins.(400% faster).   
This project is set up in a way that makes it easier to create your own ROS packages later for processing tactile data in your applications.
## Visualization
### Estimating object pose by fitting an ellipse (PCA and OpenCV):
<br />
<p align="center">
  <img src="https://github.com/vocdex/digit-depth/blob/main/assets/depthPCA.gif" width="400" title="depth">
</p>

### Depth image point cloud :
<br />
<p align="center">
  <img src="https://github.com/vocdex/digit-depth/blob/main/assets/point-cloud.gif" width="400" title="point-cloud">
</p>


### Marker movement tracking ( useful for force direction and magnitude estimation):
<br />
<p align="center">
  <img src="https://github.com/vocdex/digit-depth/blob/main/assets/markers.gif" width="400" title="marker">
</p>

## TODO
- Add a Pix2Pix model to generate depth maps from RGB images.
- Add an LSTM model for predicting slip from collected video frames.
- Add a baseline ResNet based model for estimating total normal force magnitude.
## Usage
Change **gel height,gel width, mm_to_pix, base_img_path, sensor :serial_num ** values in rgb_to_normal.yaml file in config folder.
- `pip install . `
- `cd scripts`
    - `python record.py` : Press SPACEBAR to start recording.
    - `python label_data.py` : Press LEFTMOUSE to label center and RIGHTMOUSE to label circumference.
    - `python create_image_dataset.py` : Create a dataset of images and save it to a csv file.
    - `python train_mlp.py` : Train an MLP model for RGB to Normal mapping.

color2normal model will be saved to a separate folder "models" in the same directory as this file.

For ROS, you can use the following command to run the node:
```bash
python scripts/ros/depth_value_pub.py
python scripts/ros/digit_image_pub.py
```
depth_value_pub.py publishes the maximum depth (deformation) value for the entire image when object is pressed. Accuracy depends on your MLP-depth model.
digit_image_pub.py publishes compressed RGB images from DIGIT.
### Please star this repo if you like it!
### Feel free to post an issue and create PRs.
