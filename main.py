import cv2
import numpy as np
import matplotlib.pyplot as plt
from Product_Recognition import match, denoising, plot_images, template_matching_Zncc
def main():
    MIN_MATCH_COUNT = 20
    MIN_MATCH_MASK_COUNT = 10
    MIN_MAX_VAL = 0.37
    
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Load reference images
    reference_images = []
    reference_keypoints = []
    reference_descriptors = []

    for i in range(1,15):
    #   img = cv2.imread(f'drive/MyDrive/VisionProject/images/dataset/models/ref{i}.png', cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(f'dataset/models/ref{i}.png', cv2.IMREAD_GRAYSCALE)
        reference_images.append(img)
        kp, des = sift.detectAndCompute(img, None)
        reference_keypoints.append(kp)
        reference_descriptors.append(des)

    images = []
    for i in range(1, 6):
    #   img_train = cv2.imread(f'drive/MyDrive/VisionProject/images/dataset/scenes/scene{i}.png')
        img_train = cv2.imread(f'dataset/scenes/scene{i}.png')
        img_train_d = cv2.cvtColor(denoising(img_train), cv2.COLOR_BGR2GRAY)

        kp_train = sift.detect(img_train_d)
        kp_train, des_train = sift.compute(img_train_d, kp_train)

        for k in range(len(reference_images)):
            
            good_matches = match(reference_descriptors[k], des_train)
            lgm = len(good_matches)

            if lgm >= MIN_MATCH_COUNT:
                # print(f"scene{i} in ref{k+1} - number of matches: {lgm}")

                # building the corrspondences arrays of good matches
                src_pts = np.float32([ reference_keypoints[k][m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp_train[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

                # Using RANSAC to estimate a robust homography.
                # It returns the homography M and a mask for the discarded points
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

                # Mask of discarded point used in visualization
                matchesMask = mask.ravel().tolist()
                l = sum(matchesMask)

                if l >= MIN_MATCH_MASK_COUNT:

                    # Corners of the query image
                    h,w = reference_images[k].shape
                    
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    
                    # Projecting the corners into the train image
                    dst = cv2.perspectiveTransform(pts,M)
                    
                    # Changing the size of template base on target that the algorithm found
                    dst_int = np.int32(dst)
                    x, y, w_d, h_d = cv2.boundingRect(dst_int)
                    dsize = (w_d, h_d)

                    # Resize the image
                    resized_image = cv2.resize(reference_images[k], dsize, interpolation=cv2.INTER_LINEAR)
                    
                    # Template_matching
                    max_Val = template_matching_Zncc(img_train_d, resized_image)
                    
                    if max_Val >= MIN_MAX_VAL:
                        # Drawing the bounding box
                        cv2.polylines(img_train,[np.int32(dst)],True,((9*(k + 1) % 255), 255, (5*(k + 1) % 255)),7, cv2.LINE_AA)

        images.append(cv2.cvtColor(img_train, cv2.COLOR_BGR2RGB)) 
     
    plot_images(images)

if __name__ == "__main__":
    main()