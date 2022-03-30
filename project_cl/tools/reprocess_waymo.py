import os

data_path = '/mnt/share_data/waymo/'
dataset_type_list = ['training', 'validation', 'testing']

_target_path = '/mnt/share_data/waymo_processed/'
# target_path = '/mnt/lustre/share_data/lizhenyu1/waymo/'

for dataset_type in dataset_type_list:
    dataset_path = data_path + dataset_type
    target_path = os.path.join(_target_path, dataset_type)
    folder_list = os.listdir(dataset_path)
    folder_list.remove(dataset_type + '(bak)')
    folder_list.remove(dataset_type)
    os.makedirs(target_path, exist_ok=True)

    for each_folder in folder_list:
        folder_path = os.path.join(dataset_path, each_folder)
        data_records_list = os.listdir(folder_path)
        data_records_list.remove('LICENSE')
        for data_record in data_records_list:
            print(os.path.join(folder_path, data_record))
            print(target_path)
            cmd = 'ln -s %s %s' % (os.path.join(folder_path, data_record), target_path)
            os.system(cmd)