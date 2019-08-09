# k-means image segmentation based on colors

1. K-means color segmentation in rgb color space.

Uses all the image channels to perform segmentation in rgb color space.

2. K-means color segmentation in the lab color space.

Can segment based on 'a' channel, 'ab' or all the three channels of the lab color space.
Using the LAB color space as one can easily separate out the luminance information from the colors.


Note: create an output directory to save the results.

To run -
python image_clustering.py -i image_path -c no_of_clusters -ch channel_information -cspace lab_or_rgb
