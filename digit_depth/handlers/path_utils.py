import datetime
import glob
import os


def get_save_path(seed, head: str = "models"):
    """
    Make save path for whatever we are training.
    """
    date = '{}'.format( datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') )
    print(date)
    seedstr = str(seed).zfill(3)
    suffix = "{}_{}".format(date,seedstr)
    result_path = os.path.join(head, suffix)
    return result_path


def find_recent_model(model_dir):
    model_paths = glob.glob(f"{model_dir}/*.ckpt")
    model_paths.sort(key=os.path.getmtime)
    return model_paths[-1]


def find_background_img(base_path:str):
    background_img = glob.glob(f"{base_path}/images/background.png")
    return background_img[0]