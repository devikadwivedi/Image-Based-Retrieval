# Image-Based-Retrieval

This project implements a content-based image retrieval system using color and Local Binary Patterns (LBP) histograms. It allows users to query images based on their visual content and retrieve similar images from a database.

## Technical Overview

### Color Histograms

A color histogram represents the distribution of colors in an image. The project offers three methods for color histogram calculation:
- **Grayscale with 8 Bins (gray_8):** Converts the RGB image to grayscale and divides the intensity range into 8 bins.
- **Grayscale with 256 Bins (gray_256):** Similar to the previous method but with 256 bins for finer granularity.
- **RGB Histogram (rgb):** Computes separate histograms for each channel (red, green, and blue) of the RGB image.

### Local Binary Patterns (LBP)

Local Binary Patterns are used to capture texture information in images. The project supports two methods for LBP histogram calculation:
- **Whole Image:** Computes LBP features for the entire image.
- **Grid Image:** Divides the image into grids and calculates LBP features for each grid.

### Feature Extraction

The system extracts feature vectors from images based on the selected methods for color histograms and LBP histograms. It offers flexibility in choosing between color histograms, LBP histograms, or both for feature extraction.

### Image Comparison

To compare images, the system computes the distance between their feature vectors using various distance measures such as Euclidean distance. Images with similar visual content are ranked higher in the retrieval results.

## Usage

To run the program, execute the following command in the terminal:

```
python main.py -q query_image_name
```

Additional options can be specified:
- `-f`: Specify the type of feature (color, lbp, both).
- `-color`: Specify the method for color histogram calculation (gray_8, gray_256, rgb).
- `-lbp`: Specify the method for LBP histogram calculation (whole_image, grid_image).
- `-dist`: Specify the distance measure used for image comparison (e.g., euclidean).

Example:

```
python main.py -q beach_1 -f both -color rgb -lbp whole_image -dist euclidean
```

## Example

Suppose you have a database of landscape images and want to retrieve images similar to a given query image of a beach. By running the program with appropriate arguments, you can quickly find visually similar images from the database, aiding in tasks like image search and recommendation.


## Acknowledgements
All images used in the database are sourced externally. Additionally, portions of the setup code in main.py are sourced from assignments.