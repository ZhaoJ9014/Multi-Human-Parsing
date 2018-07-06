# Annotation Tool for Multi-Human Parsing


## Pre-Requisites


* Python 2.7x


## Prepare Images for Annotation


* Put the image data to annotate under ./data/images/, the annotations will be stored under ./data/annotations/ automatically after annotation.


* Modify Generate_List.py to generate your own lists for annotations and images. Copy the contents to replace the counterparts of "annotationURLs" and "imageURLs" in "multi_person_web.json" and "multi_person.json".


* Modify the contents of "labels" in "multi_person_web.json" and "multi_person.json" based on your own semantic category definition.


## Annotation Instructions


* Run the python script "SimpleHTTPServerWithUpload.py".


* Open your explorer and input the link http://127.0.0.1:8000/ to access the annotation tool, as shown in the below figure. The left image is for reference, while the right image is for annotation.
<img src="https://github.com/ZhaoJ9014/Multi-Human-Parsing_MHP/blob/master/Annotation_Tools/Multi-Human_Parsing/Pub/SuppFig1.png" width="1000px"/>


* Use left-click to select among "Polygon tool", "Superpixel tool", and "Brush tool" at the right bottom panel.


* Use "zoom" at the top panel to zoom in / out the image for annotation.


* Use "boundary" at the top panel to tune the boundary of the superpixels generated for assisting annotation.


* Use left-click to select 1 attribute (semantic category) from the label list at the right panel to annotate each semantic category on each person in each image; 


* Left-click "undo" / "redo" to undo / redo.


* Left-click "export" at the right bottom to export the current annotation in "PNG" format.


* Left-click "Prev" / "Next" at the left top panel to access the previous / next image.


* Note: the images have been stored at ./your_AnnotationTool/data/images/; the annotations will be automatically stored at ./your_AnnotationTool/data/images/annotations/ after "export".
