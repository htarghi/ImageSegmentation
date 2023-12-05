import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class ImageSegmentation:
    def __init__(self, image_path):
        self.image_path=image_path
    def watershed_segmentation(self):# Read the image
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                raise  FileNotFoundError(f"Unable to read image at '{image_path}'") 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ## Apply thresholding to obtain a binary image
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Perform morphological operations to remove noise and smoothen edges
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Finding sure foreground area using distance transform
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Marker labelling for watershed
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(img, markers.astype(np.int32))  # Ensure 'markers' is of type np.int32

            # Mark watershed boundaries in the original image
            img[markers == -1] = [255, 0, 0]

            # Display the result
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Watershed Segmentation')
            plt.axis('off')
            plt.show()
            return img
        except Exception as e:
            print(f"Error: {e}")
    def extract_properties(self,segmented_markers):
        # Convert the segmented_markers image to grayscale if needed
        if len(segmented_markers.shape) > 2:
            segmented_markers = cv2.cvtColor(segmented_markers, cv2.COLOR_BGR2GRAY)
        # Apply connected components to get properties of segmented regions
        num_labels, markers = cv2.connectedComponents(segmented_markers)
        stats = cv2.connectedComponentsWithStats(segmented_markers, connectivity=8, ltype=cv2.CV_32S)

        # Extract properties such as area, centroid, bounding box, etc.
        areas = stats[2][:, cv2.CC_STAT_AREA]
        centroids = stats[3][:, :2]
        bounding_boxes = stats[2][:, cv2.CC_STAT_LEFT:cv2.CC_STAT_TOP + 4]  # Adjust columns based on desired properties

        # Create a Pandas DataFrame to store the properties
        data = {
            'Area': areas,
            'Centroid_X': centroids[:, 0],
            'Centroid_Y': centroids[:, 1],
            'Bounding_Box_Left': bounding_boxes[:, 0],
            'Bounding_Box_Top': bounding_boxes[:, 1],
            'Bounding_Box_Width': bounding_boxes[:, 2],
            'Bounding_Box_Height': bounding_boxes[:, 3]
        }

        df = pd.DataFrame(data)
        return df

    def write_properties_to_csv(self,properties_df,output_file):
        properties_df.to_csv(output_file, index=False)
if __name__ == "__main__":
    image_path = 'C:/Segmentation_package/data/cell_sam.png'  # Replace with the path to your image
    segmenter = ImageSegmentation(image_path)
    # Perform segmentation
    segmented_image = segmenter.watershed_segmentation()
    properties_df =segmenter.extract_properties(segmented_image)
    segmenter.write_properties_to_csv(properties_df, 'segment_properties.csv')
