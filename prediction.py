# -*- coding: utf-8 -*-
#from __future__ import print_function
import os.path as osp
import sys
import os
import numpy as np
import cv2
sys.path.insert(0,'/workspace/mnt/group/face/chendan/caffe-p100/python')
import caffe
import argparse
import time 
sys.path.append('./src')
from myDataLayer import RondomRotate,centerCrop


#attrList = ['prob_gender','prob_age', 'prob_mask', 'prob_glass', 'prob_expre', 'prob_block', 'prob_weizu']

def predict_gender_and_age_by_single_aligned_img(model_def, model_weights, testfile, imgbasePath):

    caffe.set_mode_gpu()  #设置为GPU运行 
    caffe.set_device(1)   #设置几号GPU   
    net = caffe.Net(model_def,      
                    model_weights,  
                    caffe.TEST)     

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  
    transformer.set_raw_scale('data', 255)  
    transformer.set_mean('data', np.array([127.5, 127.5, 127.5])) 
    transformer.set_input_scale('data', 1/128.0) 
    transformer.set_channel_swap('data', (2, 1, 0))  

    correct_nums = 0
    true_age_num = 0
    result_pre = open('./result_pre_all.txt','w')
    gender_fn = open('./result_gender.txt', 'w')
    nums =0
    result_pre.writelines('img' + '\t' + 'true_label' + '\t' + 'prediction' + '\n')
    with open(testfile, "r") as fr:
        for line in fr:
	    nums +=1
            temp = line.strip().split('\t')
            label = temp[2] 
	    gender_  = temp[1]
            imgPath = os.path.join(imgbasePath, temp[0])
            image = caffe.io.load_image(imgPath) 
	    image = centerCrop(image ,(112,112))
            transformed_image = transformer.preprocess('data', image)
            net.blobs['data'].data[...] = transformed_image
            net.blobs['data'].reshape(1, 3, 112, 112)
            output = net.forward()
	    arg = net.blobs["age_pre"].data[0]
	    gender_arg = net.blobs["prob_gender"].data[0]
	    gender_label = np.argmax(gender_arg)
# 	    print "img is:",temp[0]
#	    print "predict gender:",gender_label
#	    print "gender score:", gender_arg
#	    print "age is:", arg[0][0][0]
	    #print ("gender_arg:",gender_arg)
	    #print ("image {0} predicted age is {1}:".format(line.strip(), arg[0][0][0]))
	    result_pre.writelines(temp[0] + '\t' + str(gender_label) + '\t' + str(arg[0][0][0]) + '\n')
            if str(gender_label) == gender_:
               correct_nums += 1
	    else:
		gender_fn.writelines(temp[0] + '\t' + gender_  + '\t' + str(gender_label) + '\n')
	    age_pre = arg[0][0][0]
	    if (int(label) >= 18 and age_pre >= 18) or (int(label) < 18 and age_pre < 18):
		true_age_num +=1
#	    else:
#		result_pre.writelines(temp[0] + '\t' + str(arg) + '\t' + label + '\n')
    correct_nums_ratio = correct_nums/(nums+0.0)
    age_ratio = true_age_num/(nums+0.0)
    print ("gender acc is:", correct_nums_ratio)
    print ("age acc is:",age_ratio )
    gender_fn.close()
    result_pre.close()
#    return correct_nums_ratio

if __name__ == "__main__":
    model_def = "/workspace/mnt/group/face/chendan/CelebA/code/mobilenet_attributev1_age_gender0127/model/deploy.prototxt"
    model_weights = "/workspace/mnt/group/face/chendan/CelebA/code/mobilenet_attributev1_age_gender0127/save_model_10/mobilenet_age_reg_iter_40000.caffemodel"
    testfile = "/workspace/mnt/group/face/chendan/CelebA/code/mobilenet_attributev1_age_gender0127/src/test.txt"
    imgbasePath = "/workspace/mnt/group/face/chendan/CelebA/code/eight-attributes/images/images_Attribute_Aligned_200x150"
    #imgbasePath = "/workspace/mnt/group/face/chendan/CelebA/code/mobilenet_attributev1_age_gender0127/python-caffe"
    result = predict_gender_and_age_by_single_aligned_img(model_def, model_weights, testfile, imgbasePath)
