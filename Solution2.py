#!/usr/bin/env python
# coding: utf-8

# In[134]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[ ]:


filePath = input()
sigma = input()
windowSize = input()
img = plt.imread(filePath)
plt.imshow(img)


# img = plt.imread("Maestro.jpg")
# plt.imshow(img)
# windowSize = 5
# sigma = 0.9

# In[144]:


def kernelGen (sigma, windowSize):
    kernel = np.zeros((windowSize, windowSize))
    
    for i in range(-windowSize//2 , windowSize//2):
        for j in range(-windowSize//2 , windowSize//2):
            kernel[i + windowSize//2, j + windowSize//2] = (1/(2 * math.pi * sigma**2)) * math.exp(-(i**2 + j**2)/2*sigma**2)
    kernel /= kernel.sum()
    return kernel


# In[145]:


kernel = kernelGen(sigma, windowSize)


# In[147]:


def filter(img, windowSize):
    result = np.zeros_like(img)
    kernel = kernelGen(sigma, windowSize)
    
    for i in range((windowSize//2) , result.shape[0] - (windowSize//2)):
        for j in range((windowSize//2) , result.shape[1] - (windowSize//2)):
            result[i,j, 0] = np.sum(np.multiply(img[i-(windowSize//2):i+(windowSize//2)+1, j-(windowSize//2):j+(windowSize//2)+1, 0],kernel))
            result[i,j, 1] = np.sum(np.multiply(img[i-(windowSize//2):i+(windowSize//2)+1, j-(windowSize//2):j+(windowSize//2)+1, 1],kernel))
            result[i,j, 2] = np.sum(np.multiply(img[i-(windowSize//2):i+(windowSize//2)+1, j-(windowSize//2):j+(windowSize//2)+1, 2],kernel))
            p = (np.sum(np.multiply(img[i-(windowSize//2):i+(windowSize//2)+1, j-(windowSize//2):j+(windowSize//2)+1, 0],kernel)))

            
    return result          
    


# In[160]:


img2 = (filter(img , windowSize))


# In[161]:


plt.imshow(img2)


# In[162]:


fig, axs = plt.subplots(1,2)
axs[0].imshow(img)
axs[1].imshow(img2)
plt.show()

