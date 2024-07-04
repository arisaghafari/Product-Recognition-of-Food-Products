import cv2
import numpy as np
from matplotlib import pyplot as plt

def denoising(noisy_img):

  median_denoised = cv2.medianBlur(noisy_img, 5)
  # nl_means_denoised = cv2.fastNlMeansDenoisingColored(median_denoised, None, 5, 5, 5, 21)
  bilateral_denoised = cv2.bilateralFilter(median_denoised, d=7, sigmaColor=75, sigmaSpace=75)

  return bilateral_denoised

def match(des_query, des_train):
  # Defining index for approximate kdtree algorithm
  FLANN_INDEX_KDTREE = 1

  # Defining parameters for algorithm
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

  # Defining search params.
  # checks=50 specifies the number of times the trees in the index should be recursively traversed.
  # Higher values gives better precision, but also takes more time
  search_params = dict(checks = 50)

  # Initializing matcher
  flann = cv2.FlannBasedMatcher(index_params, search_params)

  # Matching and finding the 2 closest elements for each query descriptor.
  matches = flann.knnMatch(des_query,des_train,k=2)

  good_matches = []
  threshold_distance = 0.65
  for m,n in matches:
    if m.distance < threshold_distance*n.distance:
      good_matches.append(m)

  return good_matches

def plot_images(images):

    fig = plt.figure(figsize=(30 ,30))
    rows = 1
    columns = 1

    if len(images) % 2 == 0:
        columns = int(len(images) / 2)
        rows = int(len(images) / 2)
    else:
        columns = int(len(images) / 2) + 1
        rows = int(len(images) / 2) + 1


    for i in range(len(images)):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.show()

def template_matching_Zncc(image, template):
    
    h_i, w_i = image.shape
  
    x, y, w_t, h_t = cv2.boundingRect(template)

    if h_i < h_t:
        template = template[y:y+h_i, x:x+w_t]

    if w_i < w_t:
        template = template[y:y+h_t, x:x+w_i]

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    return max_val