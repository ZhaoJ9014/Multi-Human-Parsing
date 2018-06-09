import transformer as ts
from PIL import Image
import numpy as np 

def _get_scalemeanstd():
	return (1.0/255,
		np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),
		np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)))

def _get_transformer_image():
	scale, mean_, std_ = _get_scalemeanstd()
	transformers = []
	if scale > 0:
		transformers.append(ts.ColorScale(np.single(scale)))
	transformers.append(ts.ColorNormalize(mean_, std_))
	return transformers

def read_img(im_path,size,padding=False,scale=True):
	if scale:
		transformers = [ts.Scale(size, Image.CUBIC, False)]
	else:
		transformers = []
	transformers += _get_transformer_image()
	transformer = ts.Compose(transformers)
	raw_im = np.array(Image.open(im_path).convert('RGB'), np.uint8)
	if padding:
		raw_im = pad(raw_im,np.uint8,True)
	res = transformer(raw_im)
	return res

def pad(img,dt,color=True):
	a,b = img.shape[:2]
	img_size = max(a,b)
	if color>1:
		res = np.zeros([img_size,img_size,color],dt)
	elif color:
		res = np.zeros([img_size,img_size,3],dt)
	else:
		res = np.zeros([img_size,img_size],dt)
	if a<b:
		start_point = (b-a)//2
		res[start_point:start_point+a] = img
	elif a>b:
		start_point = (a - b) // 2
		res[:,start_point:start_point+b] = img
	elif a==b:
		res = img
	return res