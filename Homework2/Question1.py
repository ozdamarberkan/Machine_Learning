#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2

images = list()

def stack_gray_to_rgb(image):
    stacked_image = list()
    for i in range(3):
        stacked_image.append(image)
    stacked_image = np.array(stacked_image)
    return stacked_image


for i in range(1,878):
    temp = str(i)
    image = cv2.imread("van_gogh/Vincent_van_Gogh_{}.jpg".format(temp))
    image = np.array(image)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if(image.shape == (64,64)):
        image = stack_gray_to_rgb(image)
    images.append(image)

images = np.array(images)
images = images.astype(np.float64)
print('The images matrix has a shape of : ' +str(images.shape))


# In[2]:


images_flattened = images.reshape(877,-1,3)
print('The flattened images matrix has a shape of : ' +str(images_flattened.shape))

images_flattened_blue = images_flattened[:,:,0]
images_flattened_green = images_flattened[:,:,1]
images_flattened_red = images_flattened[:,:,2]

print('The flattened_images_red has a shape of : ' +str(images_flattened_red.shape))
print('The flattened_images_green has a shape of : ' +str(images_flattened_red.shape))
print('The flattened_images_blue has a shape of : ' +str(images_flattened_red.shape))


# In[3]:


# Part 1.1

def svd(images, k):
    #images -= np.mean(images, axis=0)
    U, Sigma, V_T = np.linalg.svd(images, full_matrices=False, compute_uv=True)
    S = np.diag(Sigma)
    
    
    principal_components = U[0:k].T.dot(S[0:k])
    singular_values_k = S[0:k] 
    
    singular_values = singular_values_k[np.where(singular_values_k > 0)]
    
    eigenvalues = (S ** 2) / ((np.shape(images)[0]) - 1)
    first_10_eig = np.sum(eigenvalues[0:10])
    normalizer_eig = np.sum(eigenvalues)
    
    pve = first_10_eig / normalizer_eig
    
    return singular_values, principal_components, pve


# In[4]:


singular_values_red, pc_red, pve_red = svd(images_flattened_red, 100)
singular_values_green, pc_green, pve_green = svd(images_flattened_green, 100)
singular_values_blue, pc_blue, pve_blue = svd(images_flattened_blue, 100)


# In[5]:


figureNum = 0
plt.figure(figureNum)
x = np.arange(1,101)
plt.bar(x, singular_values_red)
plt.title('First 100 Singular Values In Descending Order For Images(Red)')
plt.xlabel('Singular Values')
plt.show()


# In[6]:


print('Proportion of variance explained (PVE) by the first 10 principal components (Red): ' +str(pve_red))


# In[7]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,101)
plt.bar(x, singular_values_green)
plt.title('First 100 Singular Values In Descending Order For Images(Green)')
plt.xlabel('Singular Values')
plt.show()


# In[8]:


print('Proportion of variance explained (PVE) by the first 10 principal components (Green): ' +str(pve_green))


# In[9]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,101)
plt.bar(x, singular_values_blue)
plt.title('First 100 Singular Values In Descending Order For Images(Blue)')
plt.xlabel('Singular Values')
plt.show()


# In[10]:


print('Proportion of variance explained (PVE) by the first 10 principal components (Blue): ' +str(pve_blue))


# In[11]:


# Part 1.2

mean = np.mean(images, axis=0)
variance = np.var(images, axis=0)
std = np.sqrt(variance)

noise = np.random.normal(loc=mean, scale=variance, size = (877, 64, 64, 3))
noise = 0.01 * noise
noise = noise.reshape(877, -1, 3)


# In[12]:


images_flattened_red_noised = images_flattened_red[:] + noise[:,:,0]
images_flattened_green_noised = images_flattened_green[:] + noise[:,:,1]
images_flattened_blue_noised = images_flattened_blue[:] + noise[:,:,2]


# In[13]:


singular_values_red_noised, pc_red_noised, pve_red_noised = svd(images_flattened_red_noised, 100)
singular_values_green_noised, pc_green_noised, pve_green_noised = svd(images_flattened_green_noised, 100)
singular_values_blue_noised, pc_blue_noised, pve_blue_noised = svd(images_flattened_blue_noised, 100)


# In[14]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,101)
plt.bar(x, singular_values_red_noised)
plt.title('First 100 Singular Values In Descending Order For Images(Red)(Noised Data)')
plt.xlabel('Singular Values')
plt.show()


# In[15]:


print('Proportion of variance explained (PVE) by the first 10 principal components (Red)(Noised Data): ' +str(pve_red_noised))


# In[16]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,101)
plt.bar(x, singular_values_green_noised)
plt.title('First 100 Singular Values In Descending Order For Images(Green)(Noised Data)')
plt.xlabel('Singular Values')
plt.show()


# In[20]:


print('Proportion of variance explained (PVE) by the first 10 principal components (Green)(Noised Data): ' +str(pve_green_noised))


# In[18]:


figureNum += 1
plt.figure(figureNum)
x = np.arange(1,101)
plt.bar(x, singular_values_blue_noised)
plt.title('First 100 Singular Values In Descending Order For Images(Blue)(Noised Data)')
plt.xlabel('Singular Values')
plt.show()


# In[19]:


print('Proportion of variance explained (PVE) by the first 10 principal components (Blue)(Noised Data): ' +str(pve_blue_noised))

