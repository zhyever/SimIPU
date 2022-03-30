# import mmcv
# import pickle
# import os
# root = '/mnt/10-5-108-187/lizhenyu1/kitti_det/public_datalist_14/'
# f1 = open(os.path.join(root,'kitti_infos_trainval.pkl'),'rb')
# trainval_pkl = pickle.load(f1)
# f2 = open(os.path.join(root,'kitti_infos_test.pkl'),'rb')
# test_pkl = pickle.load(f2)
# f3 = open(os.path.join(root,'kitti_infos_val.pkl'),'rb')
# val_pkl = pickle.load(f3)
# f3 = open(os.path.join(root,'kitti_infos_train.pkl'),'rb')
# train_pkl = pickle.load(f3)
# print(len(trainval_pkl), len(test_pkl), len(val_pkl), len(train_pkl))

# # print(trainval_pkl[:2])

# for test in test_pkl:
#     test['image']['image_idx'] = test['image']['image_idx'] + len(trainval_pkl)

# # print(trainval_pkl[:2], trainval_pkl[-2:], test_pkl[:2])
# traintest = train_pkl + test_pkl
# print(len(traintest))

# f3 = open(os.path.join(root,'kitti_infos_traintest.pkl'),'wb')
# pickle.dump(traintest, f3)
# f3.close()

#
# kitti
# for paper
# 7481 7518 3769 3712
# 11230
#

import mmcv
import pickle
import os
root = '/mnt/share_data/waymo_dataset/kitti_format'
# f1 = open(os.path.join(root,'waymo_infos_trainval.pkl'),'rb')
# trainval_pkl = pickle.load(f1)
# f2 = open(os.path.join(root,'waymo_infos_test.pkl'),'rb')
# test_pkl = pickle.load(f2)
# f3 = open(os.path.join(root,'waymo_infos_val.pkl'),'rb')
# val_pkl = pickle.load(f3)
f3 = open(os.path.join(root,'waymo_infos_train.pkl'),'rb') # 1% -> 1580, full_data->158081
train_pkl = pickle.load(f3)
print(len(train_pkl))

# f = open(os.path.join(root,'one_percentage.pkl'),'wb')
# pickle.dump(train_pkl[:1580], f)
# f.close()

# f = open(os.path.join(root,'two_percentage.pkl'),'wb')
# pickle.dump(train_pkl[:3160], f)
# f.close()

f = open(os.path.join(root,'ablation_pretrain_full.pkl'),'wb')
pickle.dump(train_pkl[1581:], f)
f.close()
print("successful")

f = open(os.path.join(root,'ablation_pretrain_half.pkl'),'wb')
pickle.dump(train_pkl[1581:1581+78250], f)
f.close()
print("successful")

f = open(os.path.join(root,'ablation_pretrain_onefifth.pkl'),'wb')
pickle.dump(train_pkl[1581:1581+31300], f)
f.close()
print("successful")

f = open(os.path.join(root,'ablation_pretrain_oneten.pkl'),'wb')
pickle.dump(train_pkl[1581:1581+15650], f)
f.close()
print("successful")


