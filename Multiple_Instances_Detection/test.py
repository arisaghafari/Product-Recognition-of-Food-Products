import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Single_Instance_Detection.Single_Product_Recognition import *

IMAGE_INDEX = 0
KEY_INDEX = 1
DES_INDEX = 2
CENTER_INDEX = 3
V_INDEX = 4

CHANNEL_INDEX = 0

def sift_method(img):
    detector = cv2.SIFT_create()
    kp, ds = detector.detectAndCompute(img, None)
    return kp, ds

def find_matching_boxes(img_train, reference, params):

    # Parameters and their default values
    SIFT_DISTANCE_THRESHOLD = params.get('SIFT_distance_threshold', 0.5)
    BEST_MATCHES_POINTS = params.get('best_matches_points', 20)
    MIN_MATCH_COUNT = 50
    MIN_MATCH_MASK_COUNT = 7

    keypoints_r = {}
    descriptors_r = {}
    red_r, green_r, blue_r = cv2.split(reference) 

    channel_r = [red_r, green_r, blue_r]
    # define channel having all zeros
    zeros_r = np.zeros(blue_r.shape, np.uint8)
    blueRGB_r = cv2.merge([zeros_r,zeros_r, blue_r])
    greenRGB_r = cv2.merge([zeros_r,green_r,zeros_r])
    redRGB_r = cv2.merge([red_r, zeros_r,zeros_r])
    ref = [redRGB_r, greenRGB_r, blueRGB_r]

    for count, c in enumerate(channel_r): 
        keypoints_r[count], descriptors_r[count] = sift_method(c)
  

    # Initialize the detector and matcher
    bf = cv2.BFMatcher()

    matched_boxes = []
    matching_img = img_train.copy()

    flag = True
    count = 0
    while(flag):
        red_t, green_t, blue_t = cv2.split(matching_img)
        channel = [red_t, green_t, blue_t]
        zeros_t = np.zeros(blue_t.shape, np.uint8)
        blueRGB_t = cv2.merge([zeros_t,zeros_t, blue_t])
        greenRGB_t = cv2.merge([zeros_t,green_t,zeros_t])
        redRGB_t = cv2.merge([red_t, zeros_t,zeros_t])
        sc = [redRGB_t, greenRGB_t, blueRGB_t]

        for count, c in enumerate(channel):
            print("channel number : ", count)
            keypoints_s, descriptors_s = sift_method(c)
            # Matching strategy for SIFT
            matches = bf.knnMatch(descriptors_s, descriptors_r[count], k=2)
            good_matches = [m for m, n in matches if m.distance < SIFT_DISTANCE_THRESHOLD * n.distance]
            good_matches = sorted(good_matches, key=lambda x: x.distance)[:BEST_MATCHES_POINTS]


            if len(good_matches) >= MIN_MATCH_COUNT:
                print("there is enough match numbers : ", len(good_matches))    
                    # Extract location of good matches

                points1 = np.float32([keypoints_s[m.queryIdx].pt for m in good_matches])
                points2 = np.float32([keypoints_r[count][m.trainIdx].pt for m in good_matches])

                # Find homography for drawing the bounding box

                H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 2)

                matchesMask = mask.ravel().tolist()
                l = sum(matchesMask)
                print("matchmask : ", l)
                if l >= MIN_MATCH_MASK_COUNT:
                    # template_matching(points2)
                    # Transform the corners of the template to the matching points in the image
                    h, w = reference.shape[:2]
                    corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, H)
                
                    matched_boxes.append(transformed_corners)
                    # print("injam!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # You can uncomment the following lines to see the matching process
                    # Draw the bounding box
                    img1_with_box = sc[count].copy()
                    matching_result = cv2.drawMatches(img1_with_box, keypoints_s, ref[count], 
                                                  keypoints_r[count], good_matches, None, 
                                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                    cv2.polylines(matching_result, [np.int32(transformed_corners)], True, (255, 0, 0), 3, cv2.LINE_AA)
                    plt.imshow(matching_result, cmap='gray')
                    plt.show()

                    count += 1

                    # Create a mask and fill the matched area with near neighbors
                    matching_img2 = cv2.cvtColor(matching_img, cv2.COLOR_BGR2GRAY) 
                    mask = np.ones_like(matching_img2) * 255
                    cv2.fillPoly(mask, [np.int32(transformed_corners)], 0)
                    mask = cv2.bitwise_not(mask)
                    matching_img = cv2.inpaint(matching_img, mask, 3, cv2.INPAINT_TELEA)
                else:
                    print("there isn't enough match MASK numbers")
                    flag = False
                    
            else:
                print("there isn't enough match numbers : ", len(good_matches))
                # print(f"template_{key} is not there..")
                flag = False

    return matched_boxes


def plot_boxes(img1, matched_boxes):
    for box in matched_boxes:
        cv2.polylines(img1, [np.int32(box)], True, (0, 255, 0), 3, cv2.LINE_AA)
        # print("box ...................... : ",box)
        x, y, w_d, h_d = cv2.boundingRect(box)
        # cv2.putText(img1, f"refrence_{box[1]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    plt.imshow(img1)
    plt.show()

def denoise(src):
    # Create the AdaptiveManifoldFilter with required arguments
    sigma_s = 16.0  # Spatial standard deviation
    sigma_r = 0.2   # Color space standard deviation
    am_filter = cv2.ximgproc.createAMFilter(sigma_s, sigma_r, False)

    # Apply the filter
    dst = denoising1(src)
    dst = am_filter.filter(dst)

    return dst

def denoising1(noisy_img):

  median_denoised = cv2.medianBlur(noisy_img, 5)
  nl_means_denoised = cv2.fastNlMeansDenoisingColored(median_denoised, None, 5, 5, 5, 21)
  bilateral_denoised = cv2.bilateralFilter(nl_means_denoised, d=7, sigmaColor=75, sigmaSpace=75)

  return bilateral_denoised

def main():
    refrence_images_features = []


    # for i in range(15, 16):
    template = cv2.imread(f'../dataset/models/ref15.png') # Template
    #     template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    #     blue,green,red = cv2.split(template)

    #     kp = []
    #     des = []
    #     keypoints_r = []
    #     descriptors_r = []
    #     kp_temp = []
    #     des_temp = []
    #     # Find keypoints and descriptors for the template
    #     for channel in [blue, green, red]:
    #         kp, des = sift_method(channel)
    #         kp_temp.append(kp)
    #         des_temp.append(des)
        
    #     keypoints_r.append(kp_temp)
    #     descriptors_r.append(des_temp)

    #     refrence_images_features = [template, keypoints_r, descriptors_r]

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
        matched_boxes = find_matching_boxes(img1_d, template, params) 

        # Draw the bounding boxes on the original image
        
        plot_boxes(img1, matched_boxes)    
 
if __name__ == "__main__":
    main()
