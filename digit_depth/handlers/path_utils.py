import datetime
import os


def get_save_path(seed, head: str = "models"):
    """
    Make save path for whatever agent we are training.
    """
    date = '{}'.format( datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') )
    seedstr = str(seed).zfill(3)
    suffix = "{}_{}_{}_{}".format(date, seedstr)
    result_path = os.path.join(head, suffix)
    return result_path