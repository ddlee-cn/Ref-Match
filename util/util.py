from __future__ import print_function
import torch
from torch.autograd import Variable
import numpy as np
import scipy.io
from PIL import Image
import inspect, re
import numpy as np
import os
import math
import random
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import torchvision.transforms.functional as transFunc
import collections


def save_points_as_txt(points, path):
    with open(path, 'wt') as opt_file:
        for i in range(len(points)):
            opt_file.write('%i, %i\n' % (points[i][0], points[i][1]))

# migrated from main class
def combine_mask(mask_im_list):
    return Image.fromarray(sum([np.array(m) for m in mask_im_list])>0)

def concat_pts(pts_list):
    pts = []
    for p in pts_list:
        pts.extend(p)
    return pts

def clip_point(pt, max_region):
    # region: (xmin, xmax, ymin, ymax)
    x, y = pt
    x = max(max_region[0], x)
    x = min(max_region[1], x)
    y = max(max_region[2], y)
    y = min(max_region[3], y)
    return pt

def clip_region(region, max_region):
    # region: (xmin, xmax, ymin, ymax)
    return (max(region[0], max_region[0]), min(region[1], max_region[1]), max(region[2], max_region[2]), min(region[3], max_region[3]))

def spatial_distance(point_A, point_B):
    return math.pow((point_A - point_B).pow(2).sum(),0.5)

def extract_receptive_field(x, y, radius, width):
    center = [2*x, 2*y]
    top_left = [max(center[0]-radius, 0), max(center[1]-radius, 0)]
    bottom_right = [min(center[0]+radius+1, width[0]), min(center[1]+radius+1, width[1])]
    return [top_left, bottom_right]

def chkd(pt1, pt2):
    return max(abs(pt1[0]-pt2[0]), abs(pt1[1]-pt2[1]))

def chkd_vec(pt_list, pt):
    return [chkd(pt, p) for p in pt_list]

def in_roi(roi, point):
    # roi: [xmin, xmax, ymin, ymax]
    # point: [x, y]
    if (point[0] >= roi[0]) and (point[0] <= roi[1]) and (point[1] >= roi[2]) and (point[1] <= roi[3]):
        return True
    else:
        return False

def chkd_roi(pts, roi):
    dis = []
    for pt in pts:
        if in_roi(roi, pt):
            dis.append(0)
        else:
            dis.append(min(chkd(pt, [roi[0], roi[2]]), chkd(pt, [roi[0], roi[3]]), chkd(pt, [roi[1], roi[2]]), chkd(pt, [roi[1], roi[3]])))
    if len(dis) == 1:
        return dis[0]
    else:
        return np.array(dis)

def get_pts_box(pts, w, h):
    wmin, hmin = w, h
    wmax, hmax = 0, 0
    
    for i, j in pts:
        if i < wmin: wmin = i
        if i > wmax: wmax = i
        if j < hmin: hmin = j
        if j > hmax: hmax = j
    return (wmin, wmax, hmin, hmax)


def mapping_to_pairs(mapping, roi, sample=0):
    # mapping[:, :, i, j] to [[x1, y1], [x2, y2]]
    i_range_min, i_range_max = 0, mapping.size(2)-1
    j_range_min, j_range_max = 0, mapping.size(3)-1
    if roi:
        i_range_min, i_range_max, j_range_min, j_range_max = roi
    pts1, pts2 = [], []
    for i in range(i_range_min, i_range_max):
        for j in range(j_range_min, j_range_max):
            pts1.append([i, j])
            pts2.append((int(mapping[0, 0, i, j].cpu().numpy()), int(mapping[0, 1, i, j].cpu().numpy())))
    if sample > 0:
        rand_idx = random.sample(range(len(pts1)), sample)
        pts1 = [pts1[i] for i in rand_idx]
        pts2 = [pts2[i] for i in rand_idx]
    return pts1, pts2

def pairs_to_mapping(mapping_init, pts1, pts2, roi, mask_cond=None):
    # [[x1, y1], [x2, y2]] to mapping[:, :, i, j]
    mapping = mapping_init.clone()
    max_region = (0, mapping.size(2), 0, mapping.size(3))
    assert len(pts1)==len(pts2)
    for start, end in zip(pts1, pts2):
        if not in_roi(roi, start):
            continue
        # mapping out of bound
        if not in_roi([0, mapping_init.size(2), 0, mapping_init.size(3)], end):
            continue
        i, j = start[0], start[1]
        clip_end = clip_point(end, max_region)
        if mask_cond is not None:
            # only update mask_region
            if mask_cond[0, i, j] > 0:
                mapping[0, 0, i, j] = clip_end[0]
                mapping[0, 1, i, j] = clip_end[1]
        else:
            # update the whole roi region
            mapping[0, 0, i, j] = clip_end[0]
            mapping[0, 1, i, j] = clip_end[1]
    return mapping

def anchor_weight_mapping(mapping_list, weight_pts_list, roi=None, mask_cond=None, anchor_radius_ratio=0.4):
    # weighted average based on checkboard distance for mapping vector fields
    assert len(mapping_list) == len(weight_pts_list)
    mapping_shape = mapping_list[0].shape
    for m in mapping_list:
        assert m.shape==mapping_shape
    mapping = mapping_list[0].clone()
    i_range_min, i_range_max = 0, mapping_shape[2]-1
    j_range_min, j_range_max = 0, mapping_shape[3]-1
    if roi:
        i_range_min, i_range_max, j_range_min, j_range_max = roi
    for i in range(i_range_min, i_range_max):
        for j in range(j_range_min, j_range_max):
            if mask_cond is not None:
                # only update mask_region
                if mask_cond[0, i, j] > 0:
                    anchor_chkds = [chkd([i, j], w) for w in weight_pts_list]
                    # for points inside anchor radius
                    in_radius = False
                    for d, m in zip(anchor_chkds, mapping_list):
                        if d < anchor_radius_ratio * min(i_range_max-i_range_min, j_range_max-j_range_min):
                            mapping[0, 0, i, j] = m[0, 0, i, j]
                            mapping[0, 1, i, j] = m[0, 1, i, j]
                            in_radius = True
                    if in_radius:
                        continue
                    # for points in overlap of anchorss
                    dist_vec = np.array([1/(d+1) for d in anchor_chkds])
                    weight_vec = dist_vec / np.linalg.norm(dist_vec, 1)
                    mapping[0, 0, i, j] = np.sum(weight_vec * np.array([m[0, 0, i, j].item() for m in mapping_list]))
                    mapping[0, 1, i, j] = np.sum(weight_vec * np.array([m[0, 1, i, j].item() for m in mapping_list]))
            else:
                # update the whole roi region
                dist_vec = np.array([1/(chkd([i, j], w)+1) for w in weight_pts_list])
                weight_vec = dist_vec / np.linalg.norm(dist_vec, 1)
                mapping[0, 0, i, j] = np.sum(weight_vec * np.array([m[0, 0, i, j].item() for m in mapping_list]))
                mapping[0, 1, i, j] = np.sum(weight_vec * np.array([m[0, 1, i, j].item() for m in mapping_list]))
                
    return mapping

def weight_mapping(mapping_list, weight_pts_list, roi=None, mask_cond=None):
    # weighted average based on checkboard distance for mapping vector fields
    assert len(mapping_list) == len(weight_pts_list)
    mapping_shape = mapping_list[0].shape
    for m in mapping_list:
        assert m.shape==mapping_shape
    mapping = mapping_list[0].clone()
    i_range_min, i_range_max = 0, mapping_shape[2]-1
    j_range_min, j_range_max = 0, mapping_shape[3]-1
    if roi:
        i_range_min, i_range_max, j_range_min, j_range_max = roi
    for i in range(i_range_min, i_range_max):
        for j in range(j_range_min, j_range_max):
            if mask_cond is not None:
                # only update mask_region
                if mask_cond[0, i, j] > 0:
                    dist_vec = np.array([1/(chkd([i, j], w)+1) for w in weight_pts_list])
                    weight_vec = dist_vec / np.linalg.norm(dist_vec, 1)
                    mapping[0, 0, i, j] = np.sum(weight_vec * np.array([m[0, 0, i, j].item() for m in mapping_list]))
                    mapping[0, 1, i, j] = np.sum(weight_vec * np.array([m[0, 1, i, j].item() for m in mapping_list]))
            else:
                # update the whole roi region
                dist_vec = np.array([1/(chkd([i, j], w)+1) for w in weight_pts_list])
                weight_vec = dist_vec / np.linalg.norm(dist_vec, 1)
                mapping[0, 0, i, j] = np.sum(weight_vec * np.array([m[0, 0, i, j].item() for m in mapping_list]))
                mapping[0, 1, i, j] = np.sum(weight_vec * np.array([m[0, 1, i, j].item() for m in mapping_list]))
                
    return mapping
    
def avg_mapping(mapping_list, roi=None, mask_cond=None):
    # average mapping vector fields
    mapping_shape = mapping_list[0].shape
    for m in mapping_list:
        assert m.shape==mapping_shape
    mapping = mapping_list[0].clone()
    i_range_min, i_range_max = 0, mapping_shape[2]-1
    j_range_min, j_range_max = 0, mapping_shape[3]-1
    if roi:
        i_range_min, i_range_max, j_range_min, j_range_max = roi
    for i in range(i_range_min, i_range_max):
        for j in range(j_range_min, j_range_max):
            if mask_cond is not None:
                # only update mask_region
                if mask_cond[0, i, j] > 0:
                    mapping[0, 0, i, j] = np.mean([m[0, 0, i, j].item() for m in mapping_list])
                    mapping[0, 1, i, j] = np.mean([m[0, 1, i, j].item() for m in mapping_list])
            else:
                # update the whole roi region
                mapping[0, 0, i, j] = np.mean([m[0, 0, i, j].item() for m in mapping_list])
                mapping[0, 1, i, j] = np.mean([m[0, 1, i, j].item() for m in mapping_list])
                
    return mapping
        


# utils: data processing
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def read_image(path, witdh):
    I = Image.open(path).convert('RGB')
    transform = get_transform(witdh)
    return transform(I).unsqueeze(0)

def read_mask(path, width):
    I = Image.open(path).convert('L')
    # I = I.resize([width, width])
    arr = np.array(I)
    arr[arr>128] = 255
    return Image.fromarray(arr)

def gen_center_roi(h, w, roi_size):
    c = [h//2, w//2]
    m = roi_size // 2
    roi = [c[0]-m, c[0]+m, c[1]-m, c[1]+m] 
    return roi

def make_mask(mask_im):
    mask_tensor = transFunc.to_tensor(mask_im)
    return (mask_tensor>0).float()

def gen_roi_mask(h, w, roi):
    # make roi region as mask: roi[xmin, ymin, xmax, ymax]
    mask = np.ones((h, w))
    mask[roi[0]:roi[1], roi[2]:roi[3]] = 0
    mask = np.expand_dims((mask>0), -1)
    mask_map = 255*(1-mask[:, :, 0])
    mask_im = Image.fromarray(mask_map.astype(np.uint8))
    return mask_im
    
def get_box(mask, mask_value=1):
    """
    mask: PIL.Image[w, h]
    """
    w, h = mask.size
    wmin, hmin = w, h
    wmax, hmax = 0, 0
    mask_arr = np.array(mask)
    for i in range(w):
        for j in range(h):
            if mask_arr[i, j] >= mask_value:
                if i < wmin: wmin = i
                if i > wmax: wmax = i
                if j < hmin: hmin = j
                if j > hmax: hmax = j
    if (wmax-wmin) > (hmax-hmin):
        hmax = hmin + (wmax-wmin)
    elif (wmax-wmin) < (hmax-hmin):
        wmax = wmin + (hmax-hmin)
    return (wmin, wmax, hmin, hmax)

def make_roi(mask, box):
    w, h = mask.size
    roi_map = np.zeros((w, h))
    roi_map[box[0]:box[1], box[2]:box[3]] = 1
    roi_map = transFunc.to_tensor(roi_map)
    return (roi_map>0).float()

def resize_dilate(img, size, filter_size):
    img = img.resize(size)
    if filter_size > 0:
        img = dilate(img, filter_size)
    return img

def dilate(img, filter_size):
    img = img.filter(ImageFilter.MaxFilter(filter_size))
    return img


def get_transform(witdh):
    transform_list = []
    osize = [witdh, witdh]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]

    return transforms.Compose(transform_list)

# utils: visualization
def upsample_map(map_values, scale_factor, mode='nearest'):
    if scale_factor == 1:
        return map_values
    else:
        upsampler = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)
        return upsampler(Variable(map_values)).data

def downsample_map(map_values, scale_factor):
    if scale_factor == 1:
        return map_values
    else:
        d = scale_factor
        downsampler = torch.nn.AvgPool2d((d, d), stride=(d, d))
        return downsampler(Variable(map_values)).data

def tensor2im(image_tensor, imtype=np.uint8, index=0):
    image_numpy = image_tensor[index].cpu().float().numpy()
    mean = np.zeros((1,1,3))
    mean[0,0,:] = [0.485, 0.456, 0.406]
    stdv = np.zeros((1,1,3))
    stdv[0,0,:] = [0.229, 0.224, 0.225]
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * stdv + mean) * 255.0
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if image_numpy.shape[2] == 1:
        image_numpy = np.tile(image_numpy, [1,1,3])
    return image_numpy.astype(imtype)

def feature2images(feature, size=[1,1], imtype=np.uint8):
    feature_np = feature.cpu().float().numpy()
    mosaic = np.zeros((size[0]*feature_np.shape[2], size[1]*feature_np.shape[3]))
    for i in range(size[0]):
       for j in range(size[1]):
           single_feature = feature_np[0,i*size[1]+j,:,:]
           stretched_feature = stretch_image(single_feature)
           mosaic[(i*feature_np.shape[2]):(i+1)*(feature_np.shape[2]),
               j*feature_np.shape[3]:(j+1)*(feature_np.shape[3])] = stretched_feature
    mosaic = np.transpose(np.tile(mosaic, [3,1,1]), (1,2,0))
    return mosaic.astype(np.uint8)

def grad2image(grad, imtype=np.uint8):
    grad_np = grad.cpu().float().numpy()
    image = np.zeros((grad.shape[2], grad.shape[3]))
    for i in range(grad_np.shape[1]):
           image = np.maximum(image, grad_np[0,i,:,:])
    return stretch_image(image).astype(imtype)

def batch2im(images_tensor, imtype=np.uint8):
    image_numpy = images_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def map2image(values_map, imtype=np.uint8):
    image_numpy = values_map[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy = stretch_image(image_numpy)
    image_numpy = np.tile(image_numpy, [1,1,3])
    return image_numpy.astype(imtype)

def binary2color_image(binary_map, color1=[0,185,252], color2=[245,117,255], imtype=np.uint8):
    assert(binary_map.size(1)==1)
    binary_ref = binary_map[0].cpu().float().numpy()
    binary_ref = np.transpose(binary_ref, (1, 2, 0))
    binary_ref = np.tile(binary_ref, [1,1,3])
    color1_ref = np.tile(np.array(color1), [binary_map.size(2),binary_map.size(3),1])
    color2_ref = np.tile(np.array(color2), [binary_map.size(2),binary_map.size(3),1])
    color_map = binary_ref*color1_ref + (1-binary_ref)*color2_ref

    return color_map.astype(imtype)

def stretch_image(image):
    min_image = np.amin(image)
    max_image = np.amax(image)
    if max_image != min_image:
        return (image - min_image)/(max_image - min_image)*255.0
    else:
        return image

def color_map(i):
    colors = [
        [255,0,0],
        [0,255,0],
        [0,0,255],
        [128,128,0],
        [0,128,128]
    ]

    if i < 5:
        return colors[i]
    else:
        return np.random.randint(0,256,3)
    
def draw_points(img, pts):
    for i in range(len(pts)):
        color = color_map(i)
        pt = [round(pts[i][0]), round(pts[i][1])]
        try:
            img = draw_circle(img, pt, color)
        except:
            print("out of bound:", pt)
    return img

def draw_points_dot(img, pts):
    for i in range(len(pts)):
        color = color_map(i)
        pt = [round(pts[i][0]), round(pts[i][1])]
        try:
            img = draw_dots(img, pt, color)
        except:
            print("out of bound:", pt)
    return img

def draw_dots(image, center, color):
    image[round(center[0]), round(center[1]), :] = color
    return image    

def draw_circle(image, center, color,  radius = 4, border_color = [255,255,255]):
    image_p = np.pad(image, ((radius,radius),(radius,radius),(0,0)),'constant')
    center_p = [center[0]+radius, center[1]+radius]
    edge_d = math.floor((2*radius + 1)/6)
    image_p[center_p[0]-radius, (center_p[1]-edge_d):(center_p[1]+edge_d+1), :] = np.tile(border_color,[3,1])
    image_p[center_p[0]+radius, (center_p[1]-edge_d):(center_p[1]+edge_d+1), :] = np.tile(border_color,[3,1])
    for i in range(1,radius):
        image_p[center_p[0]+i, center_p[1]-radius+i-1, :] = border_color
        image_p[center_p[0]-i, center_p[1]-radius+i-1, :] = border_color
        image_p[center_p[0]+i, (center_p[1]-radius+i):(center_p[1]+radius-i+1), :] = np.tile(color, [2*(radius-i)+1,1])
        image_p[center_p[0]-i, (center_p[1]-radius+i):(center_p[1]+radius-i+1), :] = np.tile(color, [2*(radius-i)+1,1])
        image_p[center_p[0]+i, center_p[1]+radius+1-i, :] = border_color
        image_p[center_p[0]-i, center_p[1]+radius+1-i, :] = border_color

    image_p[center_p[0], center_p[1]-radius, :] = border_color
    image_p[center_p[0], (center_p[1]-radius+1):(center_p[1]+radius), :] = np.tile(color, [2*(radius-1)+1,1])
    image_p[center_p[0], center_p[1]+radius, :] = border_color

    return image_p[radius:image_p.shape[0]-radius, radius:image_p.shape[1]-radius, :]

# utils: misc
def save_final_image(image, name, save_dir):
    im_numpy = tensor2im(image)
    save_image(im_numpy, os.path.join(save_dir, name + '.png'))

def save_map_image(map_values, name, save_dir, level=0, binary_color=False):
    if level == 0:
        map_values = map_values
    else:
        scale_factor = int(math.pow(2,level-1))
        map_values = upsample_map(map_values, scale_factor)
    if binary_color==True:
        map_image = binary2color_image(map_values)
    else:
        map_image = map2image(map_values)
    save_image(map_image, os.path.join(save_dir, name + '.png'))

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_tensor_as_mat(tensor, path):
    tensor_numpy = tensor.cpu().numpy()
    print(path)
    scipy.io.savemat(path, mdict={'dna': tensor_numpy})

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# def read_mask(path):
#     image = Image.open(path)
#     np_image = np.array(image)
#     np_image = np_image[:,:,0]
#     print(np_image.shape)
#     return np.where(np_image>128, 1, 0)

# utils: mapping visualiztiong
def vis_mask(mask):
    mask_arr = np.repeat(np.expand_dims((mask.cpu().numpy().transpose(1, 2, 0)[:, :, 0]*255).astype(np.uint8), -1), 3, axis=-1)
    return mask_arr

def index_mat(n):
    l = list(range(n))
    lb = np.array(l * n)
    la = np.vstack([np.array(l)]*n).transpose().flatten()
    p = np.vstack([la, lb]).transpose().reshape(n, n, -1)
    return p

def mapping_to_uv(ann):
    ann = ann[0].cpu().float().numpy()
    ann = np.transpose(ann, (1, 2, 0))
    int_mat = index_mat(ann.shape[0])
    uv_mat = ann - int_mat
    return uv_mat

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)