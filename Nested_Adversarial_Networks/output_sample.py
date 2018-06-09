import output_model
import cv2

Image_path_list = ['a.jpg','b.jpg']
net = output_model.network()

for imgname in Image_path_list:
	maskimg, segimg, instimg = net.eval(imgname)
	# save results
	# cv2.imwrite('mask_'+imgname,maskimg)
	# cv2.imwrite('seg_'+imgname,segimg)
	# cv2.imwrite('inst_'+instname,instimg)