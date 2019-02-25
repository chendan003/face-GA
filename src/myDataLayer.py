import sys
import cv2
import re

sys.path.append('/workspace/mnt/group/face/chendan/Caffe-depthwise/caffe/python')
import caffe
import numpy as np
import random
import cPickle as pickle
import os
import os.path as osp

def centerCrop(im, crop_size):
    h,w = im.shape[:2]
    h_crop = crop_size[0]
    w_crop = crop_size[1]
    h_off,w_off=(h-h_crop)//2,(w-w_crop)//2

    im = im[h_off:h_off+h_crop, w_off:w_off+w_crop]
#    for i in range(len(pts)//2):
#        pts[2*i] -= w_off
#        pts[2*i + 1] -= h_off
#    return im, pts
    return im

def RondomRotate(image, max_angle,crop_size):
#    image = cv2.imread(image)
    h,w,_ = image.shape
    c_x =  w / 2
    c_y =  h / 2
    angle = np.random.randint(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((c_x, c_y), angle, 1)   
    roate_img = cv2.warpAffine(image, M, (w, h))
    croped_img = centerCrop(roate_img, crop_size)
    return croped_img





def mirror(img):
    img = cv2.flip(img, 1)
    return img


def illumination(img):
    r0 = random.uniform(0.0, 1.0)
    r1 = random.uniform(0.0, 1.0)
    hue = r0 * 0.1 + 1
    exposure = r1 * 0.5 + 1
    if random.uniform(0.0, 1.0) > 0.5:
        exposure = 1.0 / exposure
    if random.uniform(0.0, 1.0) > 0.5:
        hue = 1.0 / hue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2] * exposure
    v = np.clip(v, 0, 255)
    # v = v.astype(np.int32)
    h = hsv[:, :, 0] * hue
    h = np.clip(h, 0, 255)
    # h = h.astype(np.int32)

    hsv[:, :, 2] = v.astype(np.uint8)
    hsv[:, :, 0] = h.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# read img list
def readSrcFile(src_file):
        f = open(src_file, 'r')
        imgPathLabelList = []
        for line in f.readlines():
            temp = line.split("\t")
            labelList = [int(i) for i in temp[1:]]
            imgPathLabelList.append([temp[0], labelList])
        return imgPathLabelList

################################################################################
#########################Train Data Layer By Python#############################
################################################################################
class Data_Layer_train(caffe.Layer):

    def setup(self, bottom, top):
#	print 'bottom is:',bottom
        if len(top) != 3: #attribute numbers + 1 
            raise Exception("Need to define tops (data, label)")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom")

        self.mean = 127.5
        self.scale = 1 / 128.0

        params = eval(self.param_str)
        self.mirror = params["mirror"]
        self.illumination = params["illumination"]
        self.batch_size = params["batch_size"]
        self.src_file = params['src_file']
        self.basepath = params['img_basepath']
	self.crop_size = params['crop_size']
        #self.im_size = params["im_size"]    # "(200, 200)" ==> [200, 200]  str ==>list
	self.max_rotate_angle = params['max_rotate_angle']
	#print('params is:', params)
        self.imgLabelList = readSrcFile(self.src_file)
        self._cur = 0  # use this to check if we need to restart the list of images

        self.data_aug_type = ["normal"]
        if self.mirror == True:
            self.data_aug_type.append("mirror")
        if self.illumination == True:
            self.data_aug_type.append("illumination")
        if ("mirror" in self.data_aug_type) and ("illumination" in self.data_aug_type):
            self.data_aug_type.append("mirror_illumination")
        
        top[0].reshape(self.batch_size, 3, self.crop_size[0], self.crop_size[1])
        for i in xrange(1, 3):
#	    print 'top i is:',top[i]
            top[i].reshape(self.batch_size, 1)


    def reshape(self, bottom, top):
        pass


    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            im, labelList = self.load_next_image()
            top[0].data[itt, ...] = im
            for nums in xrange(2):
                top[nums+1].data[itt, ...] = labelList[nums]


    def backward(self, top, propagate_down, bottom):
        pass


    def load_next_image(self):
        # If we have finished forwarding all images, then an epoch has finished
        # and it is time to start a new one
        if self._cur == len(self.imgLabelList):
            self._cur = 0
        if self._cur == 0:
            random.shuffle(self.imgLabelList)
        img_path, labelList = self.imgLabelList[self._cur]
        self._cur += 1
        # bgr
        img_Full = os.path.join(self.basepath, img_path)
        #print img_Full
        image = cv2.imread(img_Full)
        # h = image.shape[0]
        # w = image.shape[1]
        # if h != w:
        #     raise Exception("image height not equal width")
        # if h != self.im_size:
        #     raise Exception("image height not equal the prototxt input size")
        image = self.data_augment(image)

        # normalization
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image -= self.mean
        image *= self.scale

        return image, labelList

    # mirror, illumination, mirror+illumination
    def data_augment(self, image):
	image = RondomRotate(image,self.max_rotate_angle, self.crop_size) 
        # choose a type of data augment
        idx = random.randint(0, len(self.data_aug_type) - 1)

        if self.data_aug_type[idx] == 'mirror':
            image = mirror(image)
        elif self.data_aug_type[idx] == 'illumination':
            image = illumination(image)
        elif self.data_aug_type[idx] == 'mirror_illumination':
            image = illumination(image)
            image = mirror(image)
        else:
            image = image
        return image

    

################################################################################
#########################Validation Data Layer By Python########################
################################################################################
class Data_Layer_validation(caffe.Layer):

    def setup(self, bottom, top):
        if len(top) != 8:
            raise Exception("Need to define tops (data, label)")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom")

        self.mean = 127.5
        self.scale = 1 / 128.0

        params = eval(self.param_str)
        self.mirror = params["mirror"]
        self.illumination = params["illumination"]
        self.batch_size = params["batch_size"]
        self.src_file = params['src_file']
        self.basepath = params['img_basepath']

        self.im_size = params["im_size"]    # "(200, 200)" ==> [200, 200]  str ==>list
    
        self.imgLabelList = readSrcFile(self.src_file)
        self._cur = 0  # use this to check if we need to restart the list of images

        self.data_aug_type = ["normal"]
        if self.mirror == True:
            self.data_aug_type.append("mirror")
        if self.illumination == True:
            self.data_aug_type.append("illumination")
        if ("mirror" in self.data_aug_type) and ("illumination" in self.data_aug_type):
            self.data_aug_type.append("mirror_illumination")
        
        top[0].reshape(self.batch_size, 3, self.crop_size[0], self.crop_size[1])
        for i in xrange(1, 8):
            top[i].reshape(self.batch_size, 1)


    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            im, labelList = self.load_next_image()
            top[0].data[itt, ...] = im
            for nums in xrange(7):
                top[nums+1].data[itt, ...] = labelList[nums]

            # top[0].data[itt, ...] = im
            # top[1].data[itt, ...] = label
            # top[2].data[itt, ...] = pts

    def backward(self, top, propagate_down, bottom):
        pass

    def load_next_image(self):
        # If we have finished forwarding all images, then an epoch has finished
        # and it is time to start a new one
	#print 'imgLabelList is:',imgLabelList
        if self._cur == len(self.imgLabelList):
            self._cur = 0
        if self._cur == 0:
            random.shuffle(self.imgLabelList)
        img_path, labelList = self.imgLabelList[self._cur]
        self._cur += 1
        # bgr
        image = cv2.imread(os.path.join(self.basepath, img_path))
        # h = image.shape[0]
        # w = image.shape[1]
        # if h != w:
        #     raise Exception("image height not equal width")
        # if h != self.im_size:
        #     raise Exception("image height not equal the prototxt input size")
        image = self.data_augment(image)

        # normalization
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image -= self.mean
        image *= self.scale
        #print os.path.join(self.basepath, img_path), label, pts
        
        return image, labelList

    # mirror, illumination, mirror+illumination
    def data_augment(self, image):
        # choose a type of data augment
        idx = random.randint(0, len(self.data_aug_type) - 1)

        if self.data_aug_type[idx] == 'mirror':
            image = mirror(image)
        elif self.data_aug_type[idx] == 'illumination':
            image = illumination(image)
        elif self.data_aug_type[idx] == 'mirror_illumination':
            image = illumination(image)
            image = mirror(image)
        else:
            image = image
        return image



