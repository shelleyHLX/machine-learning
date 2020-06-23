# coding: utf-8
# Author: shelley
# 2020/6/2211:24
import numpy as np
import cv2

img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(img.shape)
print(type(img))
U,s,V = np.linalg.svd(img)
# print(s[0:10])
# exit(0)
print('U',U.shape)
print('s', s.shape)
print('V', V.shape)
dia = np.zeros((img.shape[0],img.shape[1]))
dia[0,0]=s[0]

a0 = s[0]*np.matmul(U[:,0:1],V[0:1,:])
print(a0.shape)
cv2.imwrite('a1.jpg',a0)
a1 = s[1]*np.matmul(U[:,0:2],V[0:2,:])
a2 = s[2]*np.matmul(U[:,0:3],V[0:3,:])

a3 = s[3]*np.matmul(U[:,0:4],V[0:4,:])
a4 = s[4]*np.matmul(U[:,0:5],V[0:5,:])

a = a0+a1+a2+a3+a4
cv2.imwrite('a.jpg',a)

sum_a = 0
for i in range(20):
    sum_a += s[i]*np.matmul(U[:,0:i+1],V[0:i+1,:])
cv2.imwrite('sum_a.jpg',sum_a)