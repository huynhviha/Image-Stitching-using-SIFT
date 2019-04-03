import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def matching_features(img1, img2):
	sift = cv2.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	
	length1 = len(kp1)
	length2 = len(kp2)
	
	matches = []
	for i in range(length1):
		for j in range(length2):
			dist = np.linalg.norm(des1[i] - des2[j])
			if dist < 250:
				tmp = (i, j)
				matches.append(tmp)

	s = np.float32([kp1[matches[j][0]].pt for j in range(len(matches))]).reshape(-1, 1, 2)
	d = np.float32([kp2[matches[j][1]].pt for j in range(len(matches))]).reshape(-1, 1, 2)
	homo_matrix, mask = cv2.findHomography(s, d, cv2.RANSAC, 5.0)
	return homo_matrix
	
def warp_image(img1, img2, homo_matrix):
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	four_corners1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1, 1, 2)
	four_corners2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1, 1, 2)
	four_corners2 = cv2.perspectiveTransform(four_corners2, homo_matrix)
	corners = np.concatenate((four_corners1, four_corners2), axis = 0)
	[max_x, max_y] = np.int32(corners.max(axis = 0).ravel() + 0.5)
	[min_x, min_y] = np.int32(corners.min(axis = 0).ravel() - 0.5)
	t = [-min_x, -min_y]
	translate_matrix = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
	result = cv2.warpPerspective(img2, translate_matrix.dot(homo_matrix), (max_x - min_x, max_y - min_y))
	result[t[1]: h1 + t[1], t[0]: w1 + t[0]] = img1
	return result

all_img = os.listdir('img/pano/') 
src = cv2.imread('img/pano/' + all_img[0])	
for i in range(1, len(all_img)):
	dst = cv2.imread('img/pano/' + all_img[i])
	homo_matrix = matching_features(src, dst)
	src = warp_image(dst, src, homo_matrix)
result = cv2.resize(src, (0,0), fx=0.7, fy=0.7) 
cv2.imshow('result', result)

cv2.waitKey(0)
