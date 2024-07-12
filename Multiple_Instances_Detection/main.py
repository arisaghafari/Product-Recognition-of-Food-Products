import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Single_Instance_Detection.Single_Product_Recognition import *
from Multiple_Product_Recognition import *


def main():
    mean_of_model_intensities_r_g_b = {}
    refrence_images_features = {}

    detector = cv2.SIFT_create()

    for i in range(15, 28):
        template = cv2.imread(f'../dataset/models/ref{i}.png') # Template
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

        # Find keypoints and descriptors for the template
        keypoints_r, descriptors_r = detector.detectAndCompute(template, None)

        refrence_images_features[i] = [template, keypoints_r, descriptors_r]
        #Find or update the center of the model image information collected in the model features dictionary
        InstanceCenter(template,refrence_images_features, i) 

        # refrence_images.append(template)
        
        b,g,r = cv2.split(template)

        # save the mean of the intensities (divided per channel) for every model image
        mean_of_model_intensities_r_g_b[i] = [np.mean(r), np.mean(g), np.mean(b)]

    params = {
        'SIFT_distance_threshold': 0.85,
        'best_matches_points': 500
    }

    for i in range(6, 7):
        print(f".......................SCENE{i}......................\n")
        img1 = cv2.imread(f'../dataset/scenes/scene{i}.png') # Image
        img_copy = img1.copy()
        img1_d = denoising(img_copy)

        # Convert to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1_d = cv2.cvtColor(img1_d, cv2.COLOR_BGR2RGB)

        # Change to "SIFT" or "ORB" depending on your requirement
        matched_boxes = find_matching_boxes(img1_d, refrence_images_features, params) 

        # Draw the bounding boxes on the original image
        plot_boxes(img1, matched_boxes)    
 
if __name__ == "__main__":
    main()
