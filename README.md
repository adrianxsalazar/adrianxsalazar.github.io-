#Bounding box to points

This repository contains tools to transform bounding box datasets into points datasets. The point placed in the center of the bounding box. The tool comes with multiple options such as creating point dataset with just one type of label or to load point datasets. 

The datasets created by this tools are a directory that contains all the original imaes with a ".txt" file with the same name as the image. The txt contains the x,y coordinates of the elements in the dataset.

### Files in the repository

"bounding_box_to_point.py" contains the main structure of the transforming tool
"cocoparser.py" A coco dataset parser to load COCO datasets to then implement our tool

#### Examples
Image with bounding boxes
![Box_image](/images/bb.png)

Transformation of the bounding boxes to points
![Point_image](/images/pd.png)

Image with bounding boxes and points
![Box_point_image](/images/bb_pd.png)
