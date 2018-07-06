import cv2
import numpy as np 
import scipy.io as sio
import os

c0 = [-1,-1]
c1 = [-1,-1]
isDrawing = False
img0 = None
img1 = None
img2 = None
img_zoom = None
stage = 0
p20 = np.zeros([20,3]).astype(np.int32)-1
d = {}
scale = 1.0
maxheight = 1000
maxwidth = 1200
minheight = 400
minwidth = 700
zoom_pix = 40
zoom_min = 10
zoom_max = 100
mousepos = [-1,-1]


inp_path = 'data'
out_path = 'label'

textStr = ['Mark the person, or press q to quit','Mark the face','Right ankle','Right knee','Right hip',\
'Left hip','Left knee','Left ankle','Pelvis','Thorax','Upper neck','head top','Right wrist','Right elbow',\
'Right shoudler','Left shoudler','Left elbow','Left wrist','Finished, Press N to next person']

isVisible = 0
visibleStr = ['Visible','Occluded','Out of image']
personNum = 0

def get_zoom(x,y):
	global zoom_pix,img3
	z_x1, z_x2, z_y1, z_y2 = x-zoom_pix, x+zoom_pix, y-zoom_pix, y+zoom_pix
	# print(z_x1,z_x2,z_y1,z_y2)
	h,w,_ = img1.shape
	if z_x1<0:
		z_x1, z_x2 = 0, zoom_pix*2
	if z_y1<0:
		z_y1, z_y2 = 0, zoom_pix*2
	if z_x2>w:
		z_x1, z_x2 = w - zoom_pix*2, w
	if z_y2>h:
		z_y1, z_y2 = h - zoom_pix*2, h
	img3 = img1.copy()
	cv2.circle(img3,(x,y),3,(0,255,0),-1)
	img3 = img3[z_y1:z_y2,z_x1:z_x2]
	# print(img3.shape)
	img3 = cv2.resize(img3,(200,200))

def drawFunc(event,x,y,flags,param):
	global isDrawing,c0,c1,img0,img1,stage,textStr,mousepos
	if stage<len(textStr)-1:
		if event==cv2.EVENT_LBUTTONDOWN:
			isDrawing=True
			c0 = [x,y]
			c1 = [x+1,y+1]
			img1 = img0.copy()
			if stage>1:
				cv2.circle(img1,(c1[0],c1[1]),7,(0,0,255),-1)
		if event==cv2.EVENT_MOUSEMOVE:
			mousepos = [x,y]
			if isDrawing:
				c1 = [x,y]
				img1 = img0.copy()
				if stage==0:
					cv2.rectangle(img1,(c0[0],c0[1]),(c1[0],c1[1]),(0,255,0),3)
				elif stage==1:
					cv2.rectangle(img1,(c0[0],c0[1]),(c1[0],c1[1]),(0,0,255),3)
				elif stage<20:
					cv2.circle(img1,(c1[0],c1[1]),7,(0,0,255),-1)
		if event==cv2.EVENT_LBUTTONUP:
			isDrawing=False

def text(string):
	global img2
	cv2.putText(img2,string,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)

def write_text():
	text(textStr[stage])

def write_text2():
	global visibleStr,isVisible
	cv2.putText(img2,visibleStr[isVisible%3],(0,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)

def save_pt():
	global stage,p20,c0,c1,isVisible
	if stage==0:
		p20[18][0] = c0[0]
		p20[18][1] = c0[1]
		p20[18][2] = isVisible%3
		p20[19][0] = c1[0]
		p20[19][1] = c1[1]
		p20[19][2] = isVisible%3
		# print(c0,c1)
		# print(p20)
	elif stage==1:
		p20[16][0] = c0[0]
		p20[16][1] = c0[1]
		p20[16][2] = isVisible%3
		p20[17][0] = c1[0]
		p20[17][1] = c1[1]
		p20[17][2] = isVisible%3
	else:
		p20[stage-2][0] = c1[0]
		p20[stage-2][1] = c1[1]
		p20[stage-2][2] = isVisible%3
	# print(p20)

def write_20p(fname):
	global p20,personNum,d
	p20 = p20 / scale
	lst = []
	for i in range(20):
		if p20[i][2]!=0.:
			p20[i][0]=-1
			p20[i][1]=-1
		lst.append([p20[i][0],p20[i][1],p20[i][2]*scale])
	lst = np.float32(lst)
	d['person_'+str(personNum)] = lst

def main(fname):
	global stage,img0,img1,img2,img3,textStr,isVisible,personNum,c0,c1,d,inp_path,out_path,scale,zoom_pix
	img0 = cv2.imread(fname)
	h,w,_ = img0.shape
	scale = 1.0
	if h>maxheight:
		scale = maxheight/h
		if w>maxwidth:
			scale2 = maxwidth/w
			scale = np.min([scale,scale2])
	if h<minheight:
		scale = minheight/h
		if w<minwidth:
			scale2 = maxwidth/w
			scale = np.min([scale,scale2])
	print('scale:',scale)
	img0 = cv2.resize(img0,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
	img1 = img0.copy()
	img3 = img1[:zoom_pix,:zoom_pix]
	cv2.namedWindow('Image')
	cv2.setMouseCallback('Image',drawFunc)

	while True:
		img2 = img1.copy()
		write_text()
		write_text2()
		cv2.imshow('Image',img2)
		get_zoom(mousepos[0],mousepos[1])
		cv2.imshow('Image_zoom',img3)
		k = cv2.waitKey(1)
		if k==ord('e'):
			isVisible+=1
		if k==ord('c'):
			if stage<len(textStr)-1:
				save_pt()
				stage += 1
				img0 = img1.copy()
				isVisible = 0
				c0 = [-1,-1]
				c1 = [-1,-1]
		elif k==ord('r'):
			img1 = img0.copy()
		elif k==ord('n') and stage==len(textStr)-1:
			stage = 0
			write_20p(fname)
			img0 = cv2.imread(fname)
			img0 = cv2.resize(img0,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
			img1 = img0.copy()
			img3 = img1[:zoom_pix,:zoom_pix]
			personNum += 1
		elif k==ord('q'):
			sio.savemat(fname.replace(inp_path,out_path),d)
			personNum = 0
			d = {}
			break
		elif k==ord('-') and zoom_pix+2>zoom_min and zoom_pix+2<zoom_max:
			zoom_pix += 2
		elif k==ord('=') and zoom_pix-2<zoom_max and zoom_pix-2>zoom_min:
			zoom_pix -= 2


# main('test.jpg')
for file in os.listdir(inp_path):
	if file.endswith('.jpg'):
		path = os.path.join(inp_path,file)
		print(path)
		main(path)