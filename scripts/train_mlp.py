import glob
import os

import pandas as pd
from PIL import Image

columns = ['img_names', 'center_x', 'center_y', 'radius']
annot_file = "/home/shuk/digit-depth/csv/annotate.csv"
annot_dataframe = pd.read_csv(annot_file,sep=',', names=columns)
print(annot_dataframe.head())
annot_dataframe.to_csv('/home/shuk/digit-depth/csv/annotate.csv', sep=',', index=False)