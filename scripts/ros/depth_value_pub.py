""" Node to publish max depth value when gel is deformed """
import os
import hydra
import rospy
from std_msgs.msg import Float32

from src.third_party import geom_utils
from src.digit.digit_sensor import DigitSensor
from src.train.mlp_model import MLP
from src.train.prepost_mlp import *
seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

@hydra.main(config_path=f"{BASE_PATH}/config", config_name="rgb_to_normal.yaml", version_base=None)
def print_depth(cfg):
    model = torch.load(cfg.model_path).to(device)
    model.eval()
    # setup digit sensor
    digit = DigitSensor(30, "QVGA", cfg.sensor.serial_num)
    digit_call = digit()
    pub = rospy.Publisher('chatter', Float32, queue_size=1)
    rospy.init_node('depth', anonymous=True)
    try:
        while not rospy.is_shutdown():
            frame = digit_call.get_frame()
            img_np = preproc_mlp(frame)
            img_np = model(img_np).detach().cpu().numpy()
            img_np, normal_img = post_proc_mlp(img_np)

            # get gradx and grady
            gradx_img, grady_img = geom_utils._normal_to_grad_depth(img_normal=img_np, gel_width=cfg.sensor.gel_width,
                                                                    gel_height=cfg.sensor.gel_height,bg_mask=None)
            # reconstruct depth
            img_depth = geom_utils._integrate_grad_depth(gradx_img, grady_img, boundary=None, bg_mask=None,max_depth=0.0237)
            img_depth = img_depth.detach().cpu().numpy().flatten()

            # get max depth value
            max_depth = np.min(img_depth)
            rospy.loginfo(f"max:{max_depth}")
            img_depth_calibrated = np.abs((max_depth - 0.02362))

            # publish max depth value
            pub.publish(Float32(img_depth_calibrated*10000))  # convert to mm

    except KeyboardInterrupt:
        print("Shutting down")
        digit().disconnect()


if __name__ == "__main__":
    rospy.loginfo("starting...")
    print_depth()
