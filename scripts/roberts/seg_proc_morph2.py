import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import sys
import random
np.set_printoptions(threshold=sys.maxsize)

data = cv2.imread('hepg2_bf_seg.png',0)

thresh = np.zeros(data.shape)
loc = np.where(data*255>130)
thresh[loc] = 255

thresh_seg = cv2.imread('thresh_seg.png',0)


rows,cols = thresh_seg.shape

color_image = np.zeros((rows,cols,3))

label_image = label(thresh_seg)

val,count = np.unique(label_image, return_counts = True)


average_size = []
cc = 1

for k in val:
    if k != 0:
        loc = np.where(label_image == k)
        average_size.append(count[cc])
        cc += 1

average_cell_size = sum(average_size)/(len(val)-1)

average_size.sort()
median_cell_size = average_size[int(len(average_size)/2)]

temp = np.zeros(thresh_seg.shape)
kernel = np.ones((3,3))

state = False

# At this point there is a image with labaled blobs = label_image, color_image that has to be filled in the end with colored blobs, val - that is a list with labels for blobs, and count that has count of pixels for each blob

colors = ((255,0,0), (0,255,0), (0,0,255), (150,150,0), (0,150,150))
cc = 1
for k in val: # Goes through each blob in the label image
    temp_list = []
    temp = np.zeros(thresh_seg.shape)
    if k != 0:
        loc = np.where(label_image == k) # gets the pixel locations for corersponding blob
        if count[k] > median_cell_size * 1.5: # + median_cell_size*50/100: #Checks for blbos who has higher size than are 1.5 higher than median size
            #color_image[loc] = (0,0,0)# Possibly has to be enabled
            
            temp[loc] = 255 # Creates an image of a single blob which is larger than the median
            
            while True: # Enables loop which will erode the specific blob
            
                da_label = label(temp) # Every time labels the image, in case of two blobs appearing from the eroosion
                
                vval, ccount = np.unique(da_label, return_counts=True)# Marks the blobs and checks the sizes

                for z in vval:
                
                    if z != 0:
                        state = False
                        if ccount[z] < (median_cell_size*50/100):
                            state = True
                if state == True:
                    break
                    
                        
                temp = cv2.erode(temp,kernel,iterations=1)
            
            # At this point there are labels for eroded blobs or blob
            
            #loc = np.where(temp == 255)
            #color_image[loc] = (255,0,0)
# ------------------------------------------------------------------------------
            da_label = label(temp)
            vval, ccount = np.unique(da_label, return_counts=True)
            for z in vval:
                temp = np.zeros(temp.shape)
                if z != 0:
                    loc = np.where(da_label == z)
                    temp[loc] = 255
                    _,cond = np.unique(temp, return_counts=True)
                    while cond[1] < median_cell_size:
                        _,cond = np.unique(temp, return_counts=True)
                        temp = cv2.dilate(temp, kernel, iterations=1)
                        
                    llabel = label(temp)
                    ss,bb = np.unique(llabel, return_counts=True)
                    loc = np.where(llabel == ss[1])
                    color_image[loc] = colors[z]


# ------------------------------------------------------------------------------

        else:
            color_image[loc] = (255,255,255)

print(average_cell_size, median_cell_size)


color_image = color_image.astype(np.uint8)
while True:
    cv2.namedWindow('processed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.imshow('processed', color_image)
    cv2.imshow('original', thresh_seg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()    

plt.plot(val[1:len(val)], count[1:len(count)], 'bo')
plt.plot(np.array([0, len(val)]), np.array([median_cell_size * 1.5, median_cell_size * 1.5]) ,'r')
plt.xlabel('unique blobs')
plt.ylabel('blob size in pixels')
plt.grid()
plt.show()

