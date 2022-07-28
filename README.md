# GUI for DIGIT tactile sensor
This GUI allow you:
- to collect image frames from DIGIT and annotate circles in them.
- Saves the annotated frames to a csv file.
- Trains a baseline MLP model for RGB to Gradient mapping.
- Uses Fast Poisson Solver to generate depth maps in real time.
- Uses PCA and OpenCV to estimate 2D object pose and display it.

Currently, labeling circles is done manually for each sensor. It can take up to an hour for annotating 30 images.  
This GUI will replace manual labeling and model training process up to 15 mins.(400% faster). That way, depth map generation for new sensors will be much faster.
This project is set up in a way that makes it easier to create your own ROS packages for processing tactile data. 
## TODO
- Add a Pix2Pix model to generate depth maps from RGB images.
- Add an LSTM model for predicting slip from collected video frames.