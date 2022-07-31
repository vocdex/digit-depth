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

### Feel free to post an issue and create PRs.
