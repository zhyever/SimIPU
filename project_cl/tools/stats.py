import mmcv
import pickle
import os

root = '/mnt/share_data/waymo_dataset/kitti_format'
# f3 = open(os.path.join(root,'two_percentage.pkl'),'rb') # 1% -> 1580, full_data->158081
f3 = open(os.path.join(root,'one_percentage_val.pkl'),'rb')

train = pickle.load(f3)


counts = {"Car": 0, "Ped":0, "Cyc":0}

for item in train:

    mask = item['annos']['camera_id'] == 0
    label = item['annos']['name'][mask]
    
    for label_item in label:
        if label_item == "Car":
            counts["Car"]+=1

        elif label_item == "Pedestrian":
            counts["Ped"]+=1

        else:
            counts["Cyc"]+=1

# print(train[-1]['annos']['camera_id'])
# print(train[-1]['annos']['name'])
print(counts)


