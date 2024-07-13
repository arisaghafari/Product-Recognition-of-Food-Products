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
    MIN_DOUBLE_MATCH_COUNT = 215
    MIN_DOUBLE_MATCH_MASK_COUNT = 40
    MAX_DOUBLE_RATIO = 0.14

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

                # dis_sum, counter = color_matching(keypoints_s,
                #                                    value[KEY_INDEX], matchesMask, matching_img, value[IMAGE_INDEX], good_matches)
                # ratio = counter / l

                # print("distance : ", dis_sum)
                # print("ratio : ", ratio)
                
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
                        if d_good >= MIN_DOUBLE_MATCH_COUNT and d_l >= MIN_DOUBLE_MATCH_MASK_COUNT and d_ratio > MAX_DOUBLE_RATIO:

                            matched_boxes.append((transformed_corners, key))

                            # You can uncomment the following lines to see the matching process
                            # Draw the bounding box
                            img1_with_box = matching_img.copy()
                            matching_result = cv2.drawMatches(img1_with_box, keypoints_s, value[IMAGE_INDEX], value[KEY_INDEX], good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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
    kp_model_ch , des_model_ch, ch_model, img_model_ch = sift_channel_sep(model)

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
        
    # kp_ref_merge = kp_ref_ch[0] + kp_ref_ch[1] + kp_ref_ch[2]
    # kp_model_merge = kp_model_ch[0] + kp_model_ch[1] + kp_model_ch[2]
    # des_ref_merge = np.vstack((des_ref_ch[0], des_ref_ch[1], des_ref_ch[2]))
    # des_model_merge = np.vstack((des_model_ch[0], des_model_ch[1], des_model_ch[2]))

    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des_model_merge, des_ref_merge, k=2)
    # good_matches = [m for m, n in matches if m.distance < SIFT_DISTANCE_THRESHOLD * n.distance]
    # good_matches = sorted(good_matches, key=lambda x: x.distance)[:BEST_MATCHES_POINTS]
    # # match_ch[i] = good_matches
    # print("double chek matching : ", len(good_matches))
    # points1 = np.float32([kp_model_merge[m.queryIdx].pt for m in good_matches])
    # points2 = np.float32([kp_ref_merge[m.trainIdx].pt for m in good_matches])
    # # Find homography for drawing the bounding box

    # H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 2)

    # matchesMask = mask.ravel().tolist()
    # l = sum(matchesMask)
    # print("double check matchmask : ", l)
    # return len(good_matches), l

    # h, w = ref.shape[:2]
    # corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    # transformed_corners = cv2.perspectiveTransform(corners, H)
                
    # cv2.polylines(model, [np.int32(transformed_corners)], True, (0, 255, 255), 5, cv2.LINE_AA)
    # plt.imshow(model)
    # plt.show()


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

# def color_matching(key_train, key_ref, matchmask, img_train, img_ref, good_matches):
#     MAX_DIST = 34
#     print("................ATTENTION........................")
#     img_t = np.copy(img_train)
#     img_r = np.copy(img_ref)
#     img_t = denoise(img_t)
#     # img_t = cv2.cvtColor(img_train, cv2.COLOR_RGB2HSV)
#     # img_r = cv2.cvtColor(img_ref, cv2.COLOR_RGB2HSV)

# # building the corrspondences arrays of good matches
#     point_train = np.float32([ key_train[m.queryIdx].pt for m in good_matches ])
#     point_ref = np.float32([ key_ref[m.trainIdx].pt for m in good_matches ])
    
#     size_train = np.float32([key_train[m.queryIdx].size for m in good_matches])
#     size_ref = np.float32([key_ref[m.trainIdx].size for m in good_matches])

#     sum = 0
#     counter = 0
#     dis_counter = 0
#     for i in range(len(matchmask)):
#         if matchmask[i] == 1: #inlier
#             x_ref = int(point_ref[i][0])
#             y_ref = int(point_ref[i][1])
#             size_r = int(size_ref[i] * math.sqrt(2))
#             # size_r = 100
#             window_red_r = img_r[y_ref:y_ref+size_r, x_ref:x_ref+size_r, 0]
#             window_green_r = img_r[y_ref:y_ref+size_r, x_ref:x_ref+size_r, 1]
#             window_blue_r = img_r[y_ref:y_ref+size_r, x_ref:x_ref+size_r, 2]
#             # img_ref[y_ref:y_ref+size_r, x_ref:x_ref+size_r]
            
#             x_t = int(point_train[i][0])
#             y_t = int(point_train[i][1])
#             size_t = int(size_train[i] * math.sqrt(2))
#             window_red_t = img_t[y_t:y_t+size_t, x_t:x_t+size_t, 0]
#             window_green_t = img_t[y_t:y_t+size_t, x_t:x_t+size_t, 1]
#             window_blue_t = img_t[y_t:y_t+size_t, x_t:x_t+size_t, 2]

#             if window_red_r.size != 0 and window_green_r.size != 0 and window_blue_r.size != 0:
#                 if window_red_t.size != 0 and window_green_t.size != 0 and window_blue_t.size != 0:
                
#                     try:
#                         value_1 = (np.mean(window_red_r), np.mean(window_green_r), np.mean(window_blue_r))
#                     except ZeroDivisionError:
#                         print("Encountered division by zero while calculating mean value1.")
#                         return 1000000, 100000

#                     try:
#                         value_2 = (np.mean(window_red_t), np.mean(window_green_t), np.mean(window_blue_t))
#                     except ZeroDivisionError:
#                         print("Encountered division by zero while calculating mean value2.")
#                         return 1000000, 100000  

#                     dist = color_distance_compute(value_1, value_2)
#                     sum += dist
#                     dis_counter += 1
#                     print("distance : ", dist)
#                     if dist >= MAX_DIST:
#                         counter += 1
                
                    

    # print(counter)
    # return (sum/dis_counter, counter)
            

    

# def color_distance_compute(val1, val2):
#     r_1, g_1, b_1 = val1
#     r_2, g_2, b_2 = val2

#     return math.sqrt((r_1 - r_2)**2 + (g_1 - g_2)**2 + (b_1 - b_2)**2)
