# Annotation Tool for Multi-Human Parsing


## Pre-Requisites


* Python 2.7x


## Prepare Images for Annotation


* Put the image data to annotate under ./data/images/, the annotations will be stored under ./data/annotations/ automatically after annotation.


* Modify Generate_List.py to generate your own lists for annotations and images. Copy the contents to replace the counterparts of "annotationURLs" and "imageURLs" in "multi_person_web.json" and "multi_person.json".


* Modify the contents of "labels" in "multi_person_web.json" and "multi_person.json" based on your own semantic category definition.


## Annotation Instructions


* Run the python script "SimpleHTTPServerWithUpload.py".


* Open your explorer and input the link http://127.0.0.1:8000/ to access the annotation tool. 
<img src="https://github.com/ZhaoJ9014/Multi-Human-Parsing_MHP/blob/master/Annotation_Tools/Multi-Human_Parsing/Pub/SuppFig1.png" width="1000px"/>
