# Annotation Tool for Multi-Human Pose


## Pre-Requisites


* Python 3


* OpenCV


## Prepare Images for Annotation


* Create a new folder named "data" and put the image data to annotate under ./data/; create a new folder named "label" and the annotations will be stored in "mat" format under ./label/ automatically after annotation.


## Annotation Instructions


* Run the python script "Mark.py".


* Drag the rectangular to mark the bounding box for each instance in the image (Re-drag a new rectangular to re-mark).


* Press "c" to enter the next step.


* Drag the rectangular to mark the bounding box for the face of each instance (Re-drag a new rectangular to re-mark).


* Press "c" to enter the next step.
 

* Repeat the similar procedures to mark the 16 human body landmarks for each instance in the image (Use left-click to mark each key point, and re-click to re-mark. For each landmark, we define 3 flags, 1 - visible, 2 - occluded by other body parts / objects except for own clothes or shoes, 3 - out of image, press "e" to switch the flag for each landmark, default: 1, if the flag is set to 2 or 3, the corresponding coordinates will be set to -1).


* Press "n" to continue to mark the next instance; Repeat the similar procedures to mark the bounding boxes and human body landmarks untill the last instance; 


* After the annotation of the very last instance of the current, first press "n" then press "q" to finish the annotation of the current image and continue to the next image (for the last image in the "./data" folder, press "q" will close the annotation window).


* To use the zoom window function, press "-" to zoom in, press "=" to zoom out.
