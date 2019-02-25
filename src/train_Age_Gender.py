import sys
import os
sys.path.append('/workspace/mnt/group/face/chendan/caffe-v100/python')
import caffe
caffe.set_device(1)
caffe.set_mode_gpu()
#os.chdir('/workspace/data/CelebA/code/face_attr_celebA/script')
solver = caffe.SGDSolver('/workspace/mnt/group/face/chendan/CelebA/code/mobilenet_attributev1_age_gender0127/model/solver.prototxt')
#solver.net.copy_from('/workspace/mnt/group/face/chendan/CelebA/code/race/MobileNet-Caffe/org_file/mobilenet.caffemodel')
solver.solve()
