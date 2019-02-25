# -*-coding:utf-8 -*-

import os
import os.path as osp
import sys

def create_new_parse_train_Result():
    file_lst = os.listdir('./')
    obt_fn = [ii for ii in file_lst if ii.endswith('train')]
    fn_ = obt_fn[0]
    fn = open(fn_).readlines()
    new_fn = fn_ + '_new'
    with open(new_fn,'w')as lg:
	lg.writelines(fn[0].strip() + ',total_loss')
	lg.writelines('\n')

	for line in fn[1:]:
	    line = line.strip()
	    spl=line.split(',')
	    loss_lst = [float(jj) for jj in spl[3:]]
	    total_loss = sum(loss_lst)
	    lg.writelines(line + ',{}'.format(total_loss) + '\n')
	   
if __name__ =='__main__':
    create_new_parse_train_Result()

