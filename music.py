import numpy as np
import cv2
from PIL import Image
from keras.models import load_model

from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array 
from keras_preprocessing import image
import datetime
from threading import Thread
#from Spotipy import *  
#import time
import pandas as pd

music_dist={0:"songs/angry.csv",1:"songs/disgusted.csv ",2:"songs/fearful.csv",3:"songs/happy.csv",4:"songs/neutral.csv",5:"songs/sad.csv",6:"songs/surprised.csv"}
show_text=[0]
def music_rec():
	# print('---------------- Value ------------', music_dist[show_text[0]])
	df = pd.read_csv(music_dist[show_text[0]])
	df = df[['Name','Album','Artist']]
	df = df.head(10)
	return df
