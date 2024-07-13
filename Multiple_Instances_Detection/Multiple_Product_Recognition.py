import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Single_Instance_Detection.Single_Product_Recognition import *
import math

IMAGE_INDEX = 0
KEY_INDEX = 1
DES_INDEX = 2
CHANNEL_INDEX = 3
SIFT_DISTANCE_THRESHOLD = 0.85
BEST_MATCHES_POINTS = 500

def find_matching_boxes(img_train, refrence_images_features):

    # Parameters to tune    
    MIN_MATCH_COUNT = 302
    MIN_MATCH_MASK_COUNT = 10
    MIN_MAX_VAL = 0.37
    MIN_DOUBLE_MATCH_COUNT = 69
    MIN_DOUBLE_MATCH_MASK_COUNT = 29
    MAX_DOUBLE_RATIO = 0.37
    MIN_DOUBLE_MATCH_COUNT_SCOUND = 175
    MIN_DOUBLE_MATCH_MASK_COUNT_SCOUND = 55

    # Initialize the detector and matcher
    detector = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    matched_boxes = []
    for key, value in refrence_images_features.items():
        matching_img = img_train.copy()

        flag = True
        count = 0
        # prev_num = [0,0,0]
        while(flag):
            # Match descriptors
            keypoints_s, descriptors_s = detector.detectAndCompute(matching_img, None)

            # Matching strategy for SIFT
            matches = bf.knnMatch(descriptors_s, value[DES_INDEX], k=2)
            good_matches = [m for m, n in matches if m.distance < SIFT_DISTANCE_THRESHOLD * n.distance]
            good_matches = sorted(good_matches, key=lambda x: x.distance)[:BEST_MATCHES_POINTS]
            

            if len(good_matches) >= MIN_MATCH_COUNT:
                print("there is enough match numbers : ", len(good_matches))    
                # Extract location of good matches
                
                points1 = np.float32([keypoints_s[m.queryIdx].pt for m in good_matches])
                points2 = np.float32([value[KEY_INDEX][m.trainIdx].pt for m in good_matches])

                # Find homography for drawing the bounding box

                H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 2)

                matchesMask = mask.ravel().tolist()
                l = sum(matchesMask)
                
                print(f"template_{key} --> match mastk : {l}")
                if l >= MIN_MATCH_MASK_COUNT: #or dis_sum <= MAX_DISTANCE):
                    
                    print("pass the threshold ....")
                    print(f"there is {count + 1} * template_{key}")
                    # template_matching(points2)
                    # Transform the corners of the template to the matching points in the image
                    h, w = value[IMAGE_INDEX].shape[:2]
                    corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, H)

                    # Changing the size of template base on target that the algorithm found
                    dst_int = np.int32(transformed_corners)
                    x, y, w_d, h_d = cv2.boundingRect(dst_int)
                    dsize = (w_d, h_d)

                    crop = matching_img[y:y+h_d, x:x+w_d]
                    
                
                    # Resize the image
                    resized_image = cv2.resize(value[IMAGE_INDEX], dsize, interpolation=cv2.INTER_LINEAR)
                    resized_image_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                    # Template_matching
                    matching_img_gray = cv2.cvtColor(matching_img, cv2.COLOR_BGR2GRAY)
                    max_Val = template_matching_Zncc(matching_img_gray, resized_image_gray)

                    print("max_val : ", max_Val)

                    if max_Val >= MIN_MAX_VAL:
                        d_good , d_l, d_ratio = double_check_matching(value[IMAGE_INDEX], crop)
                        if (d_good >= MIN_DOUBLE_MATCH_COUNT and d_l >= MIN_DOUBLE_MATCH_MASK_COUNT):
                            if d_ratio >= MAX_DOUBLE_RATIO:
                                matched_boxes.append((transformed_corners, key))
                                count += 1
                            elif d_good >= MIN_DOUBLE_MATCH_COUNT_SCOUND and d_l >= MIN_DOUBLE_MATCH_MASK_COUNT_SCOUND:
                                matched_boxes.append((transformed_corners, key))
                                count += 1

                        # # You can uncomment the following lines to see the matching process
                        # # Draw the bounding box
                        # img1_with_box = matching_img.copy()
                        # matching_result = cv2.drawMatches(img1_with_box, keypoints_s, value[IMAGE_INDEX], value[KEY_INDEX], good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                        # cv2.polylines(matching_result, [np.int32(transformed_corners)], True, (255, 0, 0), 3, cv2.LINE_AA)
                        # plt.imshow(matching_result, cmap='gray')
                        # plt.show()

                    # Create a mask and fill the matched area with near neighbors
                    matching_img2 = cv2.cvtColor(matching_img, cv2.COLOR_BGR2GRAY) 
                    mask = np.ones_like(matching_img2) * 255
                    cv2.fillPoly(mask, [np.int32(transformed_corners)], 0)
                    mask = cv2.bitwise_not(mask)
                    matching_img = cv2.inpaint(matching_img, mask, 3, cv2.INPAINT_TELEA)
                    
                    
                else:
                    print("there isn't enough match MASK numbers : ", l)
                    print(f"template_{key} is not there..")
                    flag = False
            
            else:
                print("there isn't enough match numbers : ", len(good_matches))
                print(f"template_{key} is not there..")
                flag = False

    return matched_boxes


def plot_boxes(img1, matched_boxes):
    for box in matched_boxes:
        cv2.polylines(img1, [np.int32(box[0])], True, (0, 255, 0), 3, cv2.LINE_AA)
        # print("box ...................... : ",box)
        x, y, w_d, h_d = cv2.boundingRect(box[0])
        cv2.putText(img1, f"refrence_{box[1]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    plt.imshow(img1)
    plt.show()

def sift_channel_sep(img):
    
    red, green, blue = cv2.split(img) 

    channel = [red, green, blue]
    # define channel having all zeros
    zeros = np.zeros(blue.shape, np.uint8)
    blueRGB = cv2.merge([zeros,zeros, blue])
    greenRGB = cv2.merge([zeros,green,zeros])
    redRGB = cv2.merge([red, zeros,zeros])
    ref = [redRGB, greenRGB, blueRGB]

    keypoints = {}
    descriptors = {}

    detector = cv2.SIFT_create()

    for count, c in enumerate(channel): 
        keypoints[count], descriptors[count] =  detector.detectAndCompute(c, None)

    return keypoints, descriptors, channel, ref

def double_check_matching(ref, model):
    kp_ref_ch , des_ref_ch, ch_ref, img_ref_ch = sift_channel_sep(ref)
    model_d = denoising_CLAHE(model)
    kp_model_ch , des_model_ch, ch_model, img_model_ch = sift_channel_sep(model_d)

    bf = cv2.BFMatcher()
    good_matches_merge = []
    l = 0
    for i in range(CHANNEL_INDEX):
        matches = bf.knnMatch(des_model_ch[i], des_ref_ch[i], k=2)
        good_matches = [m for m, n in matches if m.distance < SIFT_DISTANCE_THRESHOLD * n.distance]
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:BEST_MATCHES_POINTS]
        good_matches_merge += good_matches
        points1 = np.float32([kp_model_ch[i][m.queryIdx].pt for m in good_matches])
        points2 = np.float32([kp_ref_ch[i][m.trainIdx].pt for m in good_matches])
        # Find homography for drawing the bounding box

        H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 2)

        matchesMask = mask.ravel().tolist()
        l += sum(matchesMask)

    print("double check good matches", len(good_matches_merge))
    print("double check MASK : ", l)
    ratio = l/len(good_matches_merge)
    print("double match ratio : ", ratio)
    return len(good_matches_merge), l, ratio

def denoising_CLAHE(image):
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply median blur to reduce noise
    median_blur = cv2.medianBlur(image, 5)

    # Apply Gaussian blur to smooth the image
    gaussian_blur = cv2.GaussianBlur(median_blur, (5, 5), 0)

    # Apply Non-Local Means Denoising
    nlm_denoise = cv2.fastNlMeansDenoising(gaussian_blur, None, 30, 7, 21)

    # # Apply Bilateral Filter
    # bilateral_filter = cv2.bilateralFilter(nlm_denoise, 9, 75, 75)

    # # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # clahe_image = clahe.apply(bilateral_filter)

    # # Apply adaptive thresholding to enhance text visibility
    # adaptive_threshold = cv2.adaptiveThreshold(clahe_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                        cv2.THRESH_BINARY, 11, 2)
    return nlm_denoise 
