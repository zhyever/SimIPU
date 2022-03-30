import mmcv
import pickle
import os

root = '/mnt/10-5-108-187/lizhenyu1/kitti_det/public_datalist_14/'
f1 = open(os.path.join(root,'kitti_infos_train.pkl'),'rb')
trainval_pkl = pickle.load(f1)
# print(trainval_pkl)
f2 = open(os.path.join(root,'kitti_infos_test.pkl'),'rb')
test_pkl = pickle.load(f2)
# print(test_pkl)
print(len(trainval_pkl), len(test_pkl))

for test in test_pkl:
    test['image']['image_idx'] = test['image']['image_idx'] + len(trainval_pkl)
print(trainval_pkl[:2], trainval_pkl[-2:], test_pkl[:2])
trainvaltest_pkl = trainval_pkl + test_pkl
print(len(trainvaltest_pkl))

f3 = open(os.path.join(root,'kitti_infos_trainvaltest.pkl'),'wb')
pickle.dump(trainvaltest_pkl, f3)
f3.close()

######################################################################################################
######################################################################################################

# root = '/mnt/share_data/waymo_dataset/kitti_format/'
# f1 = open(os.path.join(root,'waymo_dbinfos_train.pkl'),'rb')
# print("loading dbinfo")
# gt_base = pickle.load(f1)
# for key, val in gt_base.items():
#     print("{} : {}".format(key, len(val)))
# new_dict = {}
# new_dict["Car"] = gt_base["Car"][:1000000]
# new_dict["Pedestrian"] = gt_base["Pedestrian"][:500000]
# new_dict["Cyclist"] = gt_base["Cyclist"][:10000]

# print("load successfully")

# f3 = open(os.path.join(root,'waymo_dbinfos_train_less.pkl'),'wb')
# print("saving sampled dbinfo")
# pickle.dump(new_dict, f3)
# print("save successfully")
# f3.close()

######################################################################################################
######################################################################################################

# root = '/mnt/share_data/waymo_dataset/kitti_format/'
# f1 = open(os.path.join(root,'waymo_infos_trainval.pkl'),'rb')
# train_pkl = pickle.load(f1)
# # print(trainval_pkl)
# f2 = open(os.path.join(root,'waymo_infos_test.pkl'),'rb')
# test_pkl = pickle.load(f2)
# # print(test_pkl)
# print(len(train_pkl), len(test_pkl)) #

# for test in test_pkl:
#     test['image']['image_idx'] = test['image']['image_idx'] + len(train_pkl)
# print(train_pkl[:2], train_pkl[-2:], test_pkl[:2])
# trainvaltest_pkl = train_pkl + test_pkl
# print(len(trainvaltest_pkl))

# f3 = open(os.path.join(root,'waymo_infos_traintest.pkl'),'wb')
# pickle.dump(trainvaltest_pkl, f3)
# f3.close()

# 227715