# -*-coding:utf-8 -*-
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import sys

def count_att_num(img_with_mark_fn,att_location):
    fn = open(img_with_mark_fn).readlines()
    expression_0 = 0
    expression_1 = 0
    expression_2 = 0
    expression_3 = 0
    expression_4 = 0
    expression_5 =0
    expression_6 =0

    for line in fn[1:]:
	spl=line.strip().split()
	exp=spl[att_location]
	if exp=='0':
	    expression_0+=1
        if exp=='1':
            expression_1+=1
        if exp=='2':
            expression_2+=1
        if exp=='3':
            expression_3+=1
        if exp=='4':
            expression_4+=1
        if exp=='5':
            expression_5+=1
        if exp=='6':
            expression_6+=1

    expre_list = []
    for jj in range(7):
	exec('value=expression_{}'.format(jj))
	print value
	expre_list.append(value)    
    print '0 number is:', expression_0
    print '1 number is:', expression_1
    print '2 number is:', expression_2
    print '3 number is:', expression_3
    print '4 number is:', expression_4
    print '5 number is:', expression_5
    print '6 number is:', expression_6

    return expre_list

def draw_distribution_bar(expre_list,attribute):
    x = np.arange(len(expre_list))
    plt.bar(x, expre_list, width = 0.35, align='center', label=attribute, fc='c')#, tick_label=attribute)
    plt.legend(loc='best')
    plt.xlabel(attribute)
    plt.ylabel("Numbers of each class")
    for a,b in zip(x,expre_list):
	plt.text(a,b+0.05, '%.0f' % b, ha='center', va='bottom')
    plt.title('Distribution of {}'.format(attribute))
    plt.savefig('{}.jpg'.format(attribute))
    plt.close()

if __name__=='__main__':
    img_with_mark_fn = sys.argv[1]
    loca = sys.argv[2]
    location = int(loca)
    att = ['Sex','Age', 'Mask', 'Glasses','Expression','Forehead_block','Weizu']
    num = [2,7,2,3,2,2,2]
    print '==========> count att:', att[location-1]
    expre_list = count_att_num(img_with_mark_fn, location) 
    # if you want to drawpic
    draw_distribution_bar(expre_list[:num[location-1]],att[location-1])
