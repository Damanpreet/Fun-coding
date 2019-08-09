from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import argparse
import cv2
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import time

def main():
    ar = argparse.ArgumentParser()
    ar.add_argument("-i", "--image", help="Image path")
    ar.add_argument("-c", "--clusters", type=int, default=4, help="No of clusters in each image")
    ar.add_argument("-ch", "--channels", default='12', help="Use single/both color channels of lab color space. Accepted value: 1/12/None")
    ar.add_argument("-cspace", "--color_space", default='lab', help="Choose between lab and rgb color space.")
    args = vars(ar.parse_args())

    img = '.'.join(args["image"].split("/")[-1].split('.')[:-1])

    image = cv2.imread(args["image"])
    image = cv2.medianBlur(image, 11) # remove noise
    
    if args["color_space"]=='rgb':
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rescaled_image = image_rgb
    else:
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # choose the color channels of the lab color space
        if (args["channels"]=='12'):
            print("working on channel 1 and 2 of lab image.")
            rescaled_image = image_lab[:,:,1:]
        elif (args["channels"]=='1'):
            print("Working on channel 1 of lab image.")
            rescaled_image = image_lab[:,:,1]
            rescaled_image = rescaled_image[...,np.newaxis]
        else:
            print("Working on lab image.")
            rescaled_image = image_lab

    h, w, d = rescaled_image.shape
    rescaled_image = rescaled_image.reshape((h*w, -1))

    # K means clustering
    if args["clusters"] < 2:
        print("Warning: No of clusters should be greater than or equal to 2. Setting number of clusters to 2.")
        args["clusters"] = 2

    start_t = time.time()
    kmeans = KMeans(n_clusters = args["clusters"], n_init=30, max_iter=100, n_jobs=-1, tol=1e-1, algorithm="full")
    kmeans.fit(rescaled_image)
    duration = time.time() - start_t

    cluster_cntr = kmeans.cluster_centers_
    clustered_image = cluster_cntr[kmeans.labels_].reshape((h, w, d))
    cluster_labels = np.reshape(np.array(kmeans.labels_, dtype=np.uint8), (h, w))

    if args["color_space"]=='rgb':
        plt.imsave("output/rgb_"+img+'_SEG.png', clustered_image.astype(np.uint8))
    else:
        if args["channels"]=='12':            
            # sort the labels by area
            labels_sorted = sorted([c for c in range(args["clusters"])], key=lambda x: -np.sum(cluster_labels==x))
            
            # segmented image
            kmeans_image = np.zeros((h, w), dtype = np.uint8)
            for i, label in enumerate(labels_sorted):
                kmeans_image[cluster_labels==label] = 255/(args["clusters"]) * i            
            plt.imsave("output/lab_"+img+"_SEG.png", kmeans_image, cmap='viridis')
        else:
            plt.imsave("output/lab_"+img+'_SEG.png', np.squeeze(clustered_image).astype(np.uint8), cmap='viridis')

    print("Total time taken to run the job: ", duration)

if __name__=="__main__":
    main()
