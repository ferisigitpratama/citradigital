import numpy as np
import matplotlib.pyplot as plt
import math

img=plt.imread("avocado.jpg")

vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]
n,m,d = img.shape
vertical_edges_img = np.zeros_like(img)

for row in range(3, n-2):
	for col in range(3, m-2):
		local_pixels = img[row-1:row+2, col-1:col+2, 0]
		transformed_pixels = vertical_filter*local_pixels
		vertical_score = (transformed_pixels.sum() + 4)/8
		vertical_edges_img[row, col] = [vertical_score]*3
plt.imshow(vertical_edges_img)


horizontal_edges_img = np.zeros_like(img)
for row in range(3, n-2):
	for col in range(3, m-2):
		local_pixels = img[row-1:row+2, col-1:col+2, 0]
		transformed_pixels = vertical_filter*local_pixels
		horizontal_score = (transformed_pixels.sum() + 4)/8
		horizontal_edges_img[row, col] = [vertical_score]*3
horizontal_edges_img[750,450]
horizontal_edges_img[170,224]
plt.imshow(horizontal_edges_img)


edges_img = np.zeros_like(img)
for row in range(3, n-2):
	for col in range(3, m-2):
		local_pixels = img[row-1:row+2, col-1:col+2, 0]
		vertical_transformed_pixels = vertical_filter*local_pixels
		vertical_score = vertical_transformed_pixels.sum()/4
		horizontal_transformed_pixels = horizontal_filter*local_pixels
		horizontal_score = horizontal_transformed_pixels.sum()/4
		edge_score = (vertical_score**2 + horizontal_score**2)**.5
		edges_img[row, col] = [edge_score]*3
		edges_img = edges_img/edges_img.max()

plt.imshow(edges_img)
img=plt.imread("avocado.jpg")
plt.imshow(img)

greyImg = img.mean(axis=2, keepdims=True)/255.0
greyImg = np.concatenate([greyImg]*3, axis=2)
plt.imshow(greyImg)
edges_img = np.zeros_like(greyImg)
n,m,d = greyImg.shape

for row in range(3, n-2):
	for col in range(3, m-2):
		local_pixels = img[row-1:row+2, col-1:col+2, 0]
		vertical_transformed_pixels = vertical_filter*local_pixels
		vertical_score = vertical_transformed_pixels.sum()/4
		horizontal_transformed_pixels = horizontal_filter*local_pixels
		horizontal_score = horizontal_transformed_pixels.sum()/4
		edge_score = (vertical_score**2 + horizontal_score**2)**.5
		dges_img[row, col] = [edge_score]*3
		edges_img = edges_img/edges_img.max()

plt.imshow(edges_img)
plt.show()
