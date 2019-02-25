#-*- coding: utf-8 -*-
import os
import random
import cv2
import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
import imghdr

def generate_train_test_val_txt(img_with_attibute_file, 
			save_train_Path, save_test_Path,save_val_Path):

    img_with_att = open(img_with_attibute_file).readlines()
    img_with_att = [ll.strip() for ll in img_with_att]
    img_num = len(img_with_att)        # first line is attribute names
    random.shuffle(img_with_att)
    #  train: test: val = 8:1:1
    train_list = img_with_att[: int(img_num*0.8)]
    test_list = img_with_att[int(img_num*0.8):int(img_num*0.9)]
    val_list = img_with_att[int(img_num*0.9):]

    with open(save_train_Path,'w')as tr:
	for line in train_list:
	    tr.writelines(line)
	    tr.writelines('\n')

    with open(save_test_Path,'w')as te:
        for line in test_list:
            te.writelines(line)
            te.writelines('\n')

    with open(save_val_Path,'w')as va:
        for line in val_list:
            va.writelines(line)
            va.writelines('\n')


def count_all_images_ave_w_h(img_list_fn, img_dir):
    w_a, h_a = 0, 0
    img_nums = 0
    fail = open('failed_lst.txt','w')
    
    with open(img_list_fn, "r") as fr:
	fr = fr.readlines()
        for line in fr:
            imgPath = osp.join(img_dir, line.split(" ")[0])
	    if not osp.exists(imgPath):
		print '{} doesn\'t exist!'.format(imgPath)
	    try:
                img = cv2.imread(imgPath)
                h_a += img.shape[0]
                w_a += img.shape[1]
                img_nums += 1
	    except:
		print '{} is Null!'.format(imgPath)
		fail.writelines(line.strip() + '\n')
    fail.close()

    print("ave width: {}, ave height: {}".format(w_a*1.0/img_nums, h_a*1.0/img_nums))


def draw_distribution_bar(img_with_attibute_file):
    num_lst = [0]*4  
    category_lst = ['white', 'yellow', 'block', 'arabs']

    att_fn = open(img_with_attibute_file).readlines()
    for line in att_fn:
	line = line.strip()
	spl = line.split(' ')
	label = spl[1]
	if label=='0':
	    num_lst[0] +=1
	elif label=='1':
            num_lst[1] +=1
        elif label=='2':
            num_lst[2] +=1
        else:
            num_lst[3] +=1
    
    plt.bar(category_lst, num_lst, width = 0.35, align='center', label='race0', fc='c')#, tick_label=attribute)
    plt.legend(loc='best')
    plt.xlabel('Race')
    plt.ylabel("Numbers")
    for a,b in zip(category_lst,num_lst):
        plt.text(a,b+0.05, '%.0f' % b, ha='center', va='bottom')
    plt.title('Distribution of {}'.format('race'))
    plt.savefig('{}.jpg'.format('race_bar'))
    plt.close()




        
if __name__ == "__main__":
#    img_dir = "/workspace/mnt/group/face/chendan/CelebA/code/eight-attributes/images/images_all_v2"
    img_with_attibute_file = "/workspace/mnt/group/face/chendan/CelebA/code/mobilenet_attributev1_age_gender0127/imglst_labeled_age_gender_new000.txt"
    save_train_Path = "/workspace/mnt/group/face/chendan/CelebA/code/mobilenet_attributev1_age_gender0127/src/train.txt"
    save_val_Path = "/workspace/mnt/group/face/chendan/CelebA/code/mobilenet_attributev1_age_gender0127/src/validation.txt"
    save_test_Path = "/workspace/mnt/group/face/chendan/CelebA/code/mobilenet_attributev1_age_gender0127/src/test.txt"
    generate_train_test_val_txt(img_with_attibute_file,
                        save_train_Path, save_test_Path,save_val_Path) 
#    count_all_images_ave_w_h(img_with_attibute_file, 
#                   '/workspace/mnt/group/face/chendan/CelebA/code/race')

#    draw_distribution_bar(img_with_attibute_file)

