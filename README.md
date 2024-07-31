# Product-Recognition-of-Food-Products
## Single Instance Detection
This project involves a series of steps to detect and match reference images within training images and reference images. The process begins with loading both the training and reference images. Once loaded, the images undergo preprocessing, including denoising and conversion to grayscale to facilitate further analysis.

Key points in the images are identified using the Scale-Invariant Feature Transform (SIFT) technique, which also generates descriptors for these key points. A dictionary is maintained to track match counts for each reference image. The method iterates over each reference image, comparing descriptors from both the reference and training images to count the number of good matches. 

If the number of good matches meets the threshold, matching points are used to compute a robust homography with RANSAC, projecting the reference image corners onto the training image to create a bounding box. The reference image is then scaled to this bounding box, and template matching is performed using Zero-mean Normalized Cross-Correlation (ZNCC). If the template matching score is sufficient, match counts and bounding box information are updated, and the bounding box is labeled on the training image with the reference index.

## Multiple Instance Detection
In addition to the achievements of "Single Instance Detection", we need to identify multiple instances of the reference image. To find these additional instances, the pipeline remains largely unchanged. After identifying an instance using SIFT and obtaining its corner coordinates, we blur that area along with its surrounding neighbors. We then continue searching for key points and descriptors in a loop until we no longer satisfy our thresholds.

A challenge we faced was that SIFT operates on grayscale images and cannot distinguish between objects of the same shape but different colors. This led to incorrect bounding boxes. We experimented with various methods, such as using a color histogram and different color spaces like HSV, but these were unsuccessful. Eventually, we devised a solution: matching the detected bounding boxes with a new SIFT process on each color channel, using different thresholds. This ensured accurate identification of the reference image, and the results were added to the match_counts_multiple dictionary.
