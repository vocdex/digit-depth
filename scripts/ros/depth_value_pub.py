"""
    By default, depth image measures the distance from the camera to the gel surface in meters.
    When the gel is not deformed, the depth value is max depth.When the gel is deformed, this value gets smaller.
    This node publishes the difference betweeb min depth (max deformation) and background depth value(min deformation)
    Note that we multiply by 1000 to convert from meters to millimeters.
"""
import hydra
import rospy
from pathlib import Path
from std_msgs.msg import Float32

from digit_depth.third_party import geom_utils
from digit_depth.digit.digit_sensor import DigitSensor
from digit_depth.train.prepost_mlp import *
from digit_depth.handlers import find_recent_model

seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = Path(__file__).parent.parent.parent.resolve()


def get_depth_values(cfg,model, img_np):
    """
    Calculate the depth values for an image using the given model.

    Parameters:
    - model: PyTorch model for calculating depth values
    - img_np: NumPy array representing the image

    Returns:
    - img_depth: NumPy array of depth values for the image
    """
    img_np = preproc_mlp(img_np)
    img_np = model(img_np).detach().cpu().numpy()
    img_np, _ = post_proc_mlp(img_np)

    gradx_img, grady_img = geom_utils._normal_to_grad_depth(
        img_normal=img_np, gel_width=cfg.sensor.gel_width,
        gel_height=cfg.sensor.gel_height, bg_mask=None
    )
    img_depth = geom_utils._integrate_grad_depth(
        gradx_img, grady_img, boundary=None, bg_mask=None, max_depth=cfg.max_depth
    )
    img_depth = img_depth.detach().cpu().numpy().flatten()
    return img_depth


def publish_depth_difference(model, cfg, pub):
    """
    Publish the difference between the maximum and minimum depth values
    for an image.

    Parameters:
    - model: PyTorch model for calculating depth values
    - cfg: Configuration object
    - pub: Publisher for publishing depth difference
    """
    digit = DigitSensor(cfg.sensor.fps, cfg.sensor.resolution, cfg.sensor.serial_num)
    digit_call = digit()
    try:
        dp_zero = 0
        dp_zero_counter = 0
        while not rospy.is_shutdown():
            frame = digit_call.get_frame()
            if frame is None:
                continue

            # dp_zero is the "background" depth value.
            if dp_zero_counter < 100:
                img_depth = get_depth_values(cfg,model, frame)
                dp_zero += np.min(img_depth)
                dp_zero_counter += 1
                continue
            elif dp_zero_counter == 100:
                dp_zero = dp_zero / 100
                dp_zero_counter += 1
            img_depth = get_depth_values(cfg,model, frame)
            max_deformation = np.min(img_depth)
            print(f"Max deformation : {max_deformation}")
            actual_deformation = np.abs((max_deformation - dp_zero))
            pub.publish(Float32(actual_deformation * 1000))  # convert to mm
            rospy.loginfo(f"Published msg at {rospy.get_time()}")

    except KeyboardInterrupt:
        digit().disconnect()


@hydra.main(config_path=f"{base_path}/config", config_name="digit.yaml", version_base=None)
def main(cfg):
    model_path = find_recent_model(f"{base_path}/models")
    model = torch.load(model_path).to(device)
    model.eval()
    pub = rospy.Publisher('digit/depth/value', Float32, queue_size=1)
    rospy.init_node('depth', anonymous=True)
    publish_depth_difference(model, cfg, pub)


if __name__ == "__main__":
    rospy.loginfo("starting...")
    main()