import tkinter as tk
from tkinter import *
import imageio
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading
import os
from show import startThreads

if __name__ == '__main__':
        startThreads()
