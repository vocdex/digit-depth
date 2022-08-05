# GUI for DIGIT tactile sensor
This GUI allows you:
- Collect image frames from DIGIT and annotate circles in them.
- Save the annotated frames to a csv file.
- Train a baseline MLP model for RGB to Gradient mapping.
- Generate depth maps in real-time using fast Poisson Solver.
- Estimate 2D object pose using PCA and OpenCV built-in algorithms.

Currently, labeling circles is done manually for each sensor. It can take up to an hour for annotating 30 images.  
This GUI will replace manual labeling and model training process up to 15 mins.(400% faster).  
This project is set up in a way that makes it easier to create your own ROS packages later for processing tactile data in your applications.
## TODO
- Add a Pix2Pix model to generate depth maps from RGB images.
- Add an LSTM model for predicting slip from collected video frames.
- Add a baseline ResNet based model for estimating total normal force magnitude.
## Usage
Change base_path, gel height,gel width, mm_to_pix values in rgb_to_normal.yaml file in config folder.
- pip install .
- cd scripts
    - `python record.py` : Press SPACEBAR to start recording.
    - `python label_data.py` : Press LEFTMOUSE to label center and RIGHTMOUSE to label circumference.
    - `python create_image_dataset.py` : Create a dataset of images and save it to a csv file.
    - `python train_mlp.py` : Train a MLP model for RGB to Normal mapping.

color2normal model will be saved to a separate folder "models" in the same directory as this file.
  
### Feel free to post an issue and create PRs.
