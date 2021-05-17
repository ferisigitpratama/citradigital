
import matplotlib.pyplot as plt 
import matplotlib.image as mimg
import math
import numpy as np

img=plt.imread("quizFX.jpg")
gray = np.zeros(img.shape, dtype=np.uint8)
blur = np.zeros(img.shape, dtype=np.uint8)

for y in range(img.shape[0]):
	for x in range (img.shape[1]):
		tmp=np.mean(img[y,x])
		gray[y,x]=[tmp,tmp,tmp]

for y in range (1, img.shape[0]-1):
	for x in range(1, img.shape[1]-1):
		blur[y,x]=((gray[y-1,x-1]*(1/9))+(gray[y-1,x]*(1/9))+(gray[y-1,x+1]*(1/9))
			+(gray[y,x-1]*(1/9))+(gray[y,x]*(1/9))+(gray[y,x+1]*(1/9))+(gray[y+1,x-1]*(1/9))
			+(gray[y+1,x]*(1/9))+(gray[y+1,x+1]*(1/9)))

gabung=np.hstack((gray,blur))
plt.imshow(gabung)
plt.show()









