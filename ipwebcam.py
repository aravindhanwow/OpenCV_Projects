import urllib
import cv2
import numpy as np

url = "http://192.168.0.101:8080/shot.jpg?rnd=932497"
imgResp = urllib.urlopen(url)
imgRp = np.array(bytearray(imgResp.read()),dtype=np.unit8)

