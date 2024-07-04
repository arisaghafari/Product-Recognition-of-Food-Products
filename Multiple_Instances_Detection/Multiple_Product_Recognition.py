import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Single_Instance_Detection.Single_Product_Recognition import *
import math

def find_matching_boxes(image, refrence_images, params):

    MIN_MATCH_COUNT = 260
    MIN_MATCH_MASK_COUNT = 10
    # Parameters and their default values
    SIFT_DISTANCE_THRESHOLD = params.get('SIFT_distance_threshold', 0.5)
    BEST_MATCHES_POINTS = params.get('best_matches_points', 20)

    # Initialize the detector and matcher
    detector = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    matched_boxes = []
    for i in range(len(refrence_images)):
        matching_img = image.copy()
        # Find keypoints and descriptors for the template
        keypoints2, descriptors2 = detector.detectAndCompute(refrence_images[i], None)

        flag = True
        count = 0
        while(flag):
            # Match descriptors
            keypoints1, descriptors1 = detector.detectAndCompute(matching_img, None)
            # Matching strategy for SIFT
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = [m for m, n in matches if m.distance < SIFT_DISTANCE_THRESHOLD * n.distance]
            good_matches = sorted(good_matches, key=lambda x: x.distance)[:BEST_MATCHES_POINTS]
            
            if len(good_matches) >= MIN_MATCH_COUNT:
                print("there is enough match numbers : ", len(good_matches))    
                print(f"there is {count + 1} * template_{i + 15}")
                # Extract location of good matches
                points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
                points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

                # Find homography for drawing the bounding box

                H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 2)

                matchesMask = mask.ravel().tolist()
                
                thre = color_matching(keypoints1, keypoints2, matchesMask, matching_img, refrence_images[i])
                print("threshold : ", thre)
                l = sum(matchesMask)
                print("match mastk : ", l)
                if l >= MIN_MATCH_MASK_COUNT:
                    # template_matching(points2)
                    # Transform the corners of the template to the matching points in the image
                    h, w = refrence_images[i].shape[:2]
                    corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, H)
                    matched_boxes.append(transformed_corners)

                    # You can uncomment the following lines to see the matching process
                    # Draw the bounding box
                    img1_with_box = matching_img.copy()
                    matching_result = cv2.drawMatches(img1_with_box, keypoints1, refrence_images[i], keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    cv2.polylines(matching_result, [np.int32(transformed_corners)], True, (255, 0, 0), 3, cv2.LINE_AA)
                    plt.imshow(matching_result, cmap='gray')
                    plt.show()

                    # Create a mask and fill the matched area with near neighbors
                    matching_img2 = cv2.cvtColor(matching_img, cv2.COLOR_BGR2GRAY) 
                    mask = np.ones_like(matching_img2) * 255
                    cv2.fillPoly(mask, [np.int32(transformed_corners)], 0)
                    mask = cv2.bitwise_not(mask)
                    matching_img = cv2.inpaint(matching_img, mask, 3, cv2.INPAINT_TELEA)

                    count += 1
                else:
                    print("there isn't enough match MASK numbers : ", l)
                    print(f"template_{i + 15} is not there..")
                    flag = False
            
            else:
                print("there isn't enough match numbers : ", len(good_matches))
                print(f"template_{i + 15} is not there..")
                flag = False

    return matched_boxes


def plot_boxes(img1, matched_boxes):
    for box in matched_boxes:
        cv2.polylines(img1, [np.int32(box)], True, (0, 255, 0), 3, cv2.LINE_AA)
    
    plt.imshow(img1)
    plt.show()

def old_denoise(src):
    dn = cv2.medianBlur(src, 5)
    # dn = cv.bilateralFilter(dn, 9, 75, 75)
    return dn

def denoise(src):
    # Create the AdaptiveManifoldFilter with required arguments
    sigma_s = 16.0  # Spatial standard deviation
    sigma_r = 0.2   # Color space standard deviation
    am_filter = cv2.ximgproc.createAMFilter(sigma_s, sigma_r, False)

    # Apply the filter
    dst = old_denoise(src)
    dst = am_filter.filter(dst)

    return dst

def color_matching(key_train, key_ref, matchmask, img_train, img_ref):
    print("................ATTENTION........................")
    img_t = cv2.cvtColor(img_train, cv2.COLOR_RGB2HSV)
    img_r = cv2.cvtColor(img_ref, cv2.COLOR_RGB2HSV)

    

def color_distance_compute(pxls1, pxls2):
    pass
