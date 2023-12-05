# Image Segmentation Package

This package contains utilities for performing image segmentation using the watershed algorithm.

## Installation

You can install this package via pip:


pip install git+https://github.com/htargi/ImageSegmentation.git

## Requirements

See `requirements.txt` for the list of dependencies.

# Usage
from ImageSegmentation.segmentation_module import ImageSegmentation

#Initialize the segmentation object
segmenter = ImageSegmentation('path_to_your_image.png')

#Perform image segmentation
segmented_image = segmenter.watershed_segmentation()

#Extract properties and write to CSV
properties_df = segmenter.extract_properties(segmented_image)
segmenter.write_properties_to_csv(properties_df, 'segment_properties.csv')

# Contributors

Hajar Moradmand (moradmand90@gmail.com)

# Licence
This project is licensed under the MIT License - see the LICENSE file for details.

