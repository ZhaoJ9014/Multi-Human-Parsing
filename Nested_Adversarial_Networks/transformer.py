"""
transformers for images as numpy arrays (h,w,3) using PIL
"""
import numpy as np
import numpy.random as npr
from PIL import Image

from util import as_list, get_interp_method


def Compose(transformers, multi_inputs=False):
    if not multi_inputs:
        def _impl(data):
            for ts in transformers:
                if ts is None:
                    continue
                data = ts(data)
            return data
    else:
        def _impl(*data):
            for ts in transformers:
                if ts is None:
                    continue
                data = ts(*data)
            return data
    return _impl

def MultiInputs(transformer):
    def _impl(*data):
        out = []
        for datum in data:
            out.append(transformer(datum))
        return tuple(out)
    return _impl

def ListInput(transformer):
    def _impl(data):
        out = []
        for datum in as_list(data):
            out.append(transformer(datum))
        return out
    return _impl

def MultiInputsUnpack(generator):
    def _impl(*data):
        ts = generator(data[0])
        out = []
        if type(ts) == tuple:
            for i,datum in enumerate(data):
                out.append(ts[i % len(ts)](datum))
        else:
            for datum in data:
                out.append(ts(datum))
        return tuple(out)
    return _impl

def MultiTransformers(*transformers):
    def _impl(*data):
        out = []
        for i,datum in enumerate(data):
            if transformers[i] is not None:
                out.append(transformers[i](datum))
            else:
                out.append(datum)
        return tuple(out)
    return _impl

def Bound(lower=None, upper=None):
    def _impl(data):
        if lower is not None:
            data = np.maximum(lower, data, out=data)
        if upper is not None:
            data = np.minimum(upper, data, out=data)
        return data
    return _impl

def ColorScale(scale):
    def _impl(data):
        data = data.astype(type(scale), copy=False)
        data *= scale
        return data
    return _impl

def ColorNormalize(mean, std=None):
    def _impl(data):
        data = data.astype(mean.dtype, copy=False)
        data -= mean
        if std is not None:
            data /= std
        return data
    return _impl

# Scale the smaller edge to size
def Scale(size=256, interpolation=Image.CUBIC, shorter_side=True):
    func = min if shorter_side else max
    def _impl(data):
        h, w = data.shape[:2]
        if func(h, w) == size:
            return data
        if func(h, w) == w:
            nw = size
            nh = int(np.round(1. * size * h / w))
        else:
            nw = int(np.round(1. * size * w / h))
            nh = size
        if interpolation != Image.NEAREST:
            interp_method = get_interp_method(h, w, nh, nw)
        return np.array(Image.fromarray(data.astype(np.uint8, copy=False)).resize((nw, nh), interp_method))
    return _impl

# Crop into centered square
def CenterCrop(size=224):
    def _impl(data):
        h, w = data.shape[:2]
        y0 = int(np.ceil((h - size) / 2.))
        x0 = int(np.ceil((w - size) / 2.))
        return data[y0:y0+size, x0:x0+size]
    return _impl

# Three crops
def ThreeCrops(size=224, flip=False):
    def _impl(data):
        h, w = data.shape[:2]
        assert h >= size and w >= size
        y0 = int(np.ceil((h - size) / 2.))
        x0 = int(np.ceil((w - size) / 2.))
        if h > w:
            ys = [0, y0, h-size]
            xs = [x0, x0, x0]
        else:
            ys = [y0, y0, y0]
            xs = [0, x0, w-size]
        return [data[y:y+size, x:x+size] for y, x in zip(ys, xs)]
    return _impl

# Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
def RandomSizedCrop(size=224, area_range=(0.08, 1.0), aspect_range=(3./4, 4./3), interpolation=Image.CUBIC):
    assert area_range is not None
    def _impl(data):
        imh, imw = data.shape[:2]
        area = imh * imw
        for _ in xrange(10):
            target_area = npr.uniform(*area_range) * area
            if aspect_range is not None:
                aspect_ratio = npr.uniform(*aspect_range)
                h = int(np.round(np.sqrt(target_area / aspect_ratio)))
                w = int(np.round(np.sqrt(target_area * aspect_ratio)))
                if npr.randint(2) == 0:
                    h, w = w, h
            else:
                h = w = int(np.sqrt(target_area))
            if h <= imh and w <= imw:
                y0 = npr.randint(imh - h + 1)
                x0 = npr.randint(imw - w + 1)
                crop = data[y0:y0+h, x0:x0+w, :].astype(np.uint8, copy=False)
                if interpolation != Image.NEAREST:
                    interp_method = get_interp_method(h, w, size, size)
                return np.array(Image.fromarray(crop).resize((size, size), interp_method))
        # fallback
        scale = Scale(size, interpolation)
        crop = CenterCrop(size)
        return crop(scale(data)) 
    return _impl

def GenRandomSizedCrop(size=224, area_range=(0.08, 1.0), aspect_range=(3./4, 4./3), interpolation=Image.CUBIC):
    assert area_range is not None
    def _generator(data):
        # need y0, x0, h, w, size, interpolation
        def _gen_impl(interpolation):
            def _impl(data):
                crop = data[y0:y0+h, x0:x0+w].astype(np.uint8, copy=False)
                if interpolation != Image.NEAREST:
                    interp_method = get_interp_method(h, w, size, size)
                return np.array(Image.fromarray(crop).resize((size, size), interp_method))
            return _impl
        
        imh, imw = data.shape[:2]
        area = imh * imw
        for _ in xrange(10):
            target_area = npr.uniform(*area_range) * area
            if aspect_range is not None:
                aspect_ratio = npr.uniform(*aspect_range)
                h = int(np.round(np.sqrt(target_area / aspect_ratio)))
                w = int(np.round(np.sqrt(target_area * aspect_ratio)))
                if npr.randint(2) == 0:
                    h, w = w, h
            else:
                h = w = int(np.sqrt(target_area))
            if h <= imh and w <= imw:
                y0 = npr.randint(imh - h + 1)
                x0 = npr.randint(imw - w + 1)
                if type(interpolation) == tuple:
                    out = []
                    for this_interp in interpolation:
                        out.append(_gen_impl(this_interp))
                    return tuple(out)
                else:
                    return _gen_impl(interpolation)
        
        # fallback
        if type(interpolation) == tuple:
            out = []
            for this_interp in interpolation:
                out.append(Compose([Scale(size, this_interp), CenterCrop(size)]))
            return tuple(out)
        else:
            return Compose([Scale(size, interpolation), CenterCrop(size)])
    return _generator

# Random flip with a probabily of 50%
def HorizontalFlip():
    def _impl(data):
        if npr.randint(2) == 0:
            return data[:, ::-1]
        else:
            return data
    return _impl

def GenHorizontalFlip():
    def _generator(data):
        # need flip
        def _impl(data):
            if flip:
                return data[:, ::-1]
            else:
                return data
        
        flip = npr.randint(2) == 0
        return _impl
    return _generator

# Lighting noise (AlexNet-style PCA-based noise)
def Lighting(alpha_std, eig_val, eig_vec):
    def _impl(data):
        if alpha_std == 0:
            return data
        data = data.astype(np.single, copy=False)
        alpha = npr.normal(0, alpha_std, 3)
        rgb = (eig_vec * alpha.reshape((1,3)) * eig_val.reshape((1,3))).sum(1).astype(np.single)
        data += rgb.reshape((1,1,3))
        data /= 1. + alpha_std
        return data
    return _impl

# Blend the two and save into data1
def _blend(data1, data2, alpha):
    data1 *= alpha
    data1 += data2 * (1 - alpha)
    return data1

# Transform data into dst as gray-scale format
def _gray_scale(dst, data):
    dst[:] = 0
    dst[:, :, 0] = (data * np.array([0.299, 0.587, 0.114]).reshape((1,1,3))).sum(2)
    return dst

def Saturation(size=224, var=0.4):
    gs = np.zeros((size, size, 1), np.single)
    def _impl(data):
        _gray_scale(gs, data)
        alpha = 1.0 + npr.uniform(-var, var)
        data = _blend(data.astype(np.single, copy=False), gs, alpha)
        return data
    return _impl

def Brightness(size=224, var=0.4):
    def _impl(data):
        alpha = 1.0 + npr.uniform(-var, var)
        data = _blend(data.astype(np.single, copy=False), 0, alpha)
        return data
    return _impl

def Contrast(size=224, var=0.4):
    gs = np.zeros((size, size, 1), np.single)
    def _impl(data):
        _gray_scale(gs, data)
        alpha = 1.0 + npr.uniform(-var, var)
        data = _blend(data.astype(np.single, copy=False), gs.mean(), alpha)
        return data
    return _impl

def RandomOrder(ts):
    def _impl(data):
        order = npr.permutation(len(ts))
        for i in order:
            data = ts[i](data)
        return data
    return _impl

# Andre Howard
def ColorJitter(size=224, brightness=0, contrast=0, staturation=0):
    ts = []
    if brightness > 0:
        ts.append(Brightness(size, brightness))
    if contrast > 0:
        ts.append(Contrast(size, contrast))
    if staturation > 0:
        ts.append(Saturation(size, staturation))
    if len(ts) == 0:
        return lambda data: data
    return RandomOrder(ts)

