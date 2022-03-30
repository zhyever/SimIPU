# Lint as: python3
# Copyright 2020 The Waymo Open Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================*/
"""A simple example to generate a file that contains serialized Objects proto."""

import time

import os
import time
import glob
import cv2
import pickle as pkl
import numpy as np
import tensorflow as tf

from queue import Queue
from threading import Thread


from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

from mmdet3d.core.bbox import box_np_ops

import copy

open_dataset_label = open_dataset.waymo__open__dataset_dot_label__pb2

type_dict = {
	'TYPE_UNKNOWN': 0,
	'TYPE_VEHICLE': 1,
	'TYPE_PEDESTRIAN': 2,
	'TYPE_SIGN': 3,
	'TYPE_CYCLIST': 4,
}
camera_dict = {
	'UNKNOWN': 0,
	'FRONT': 1,
	'FRONT_LEFT':2,
	'FRONT_RIGHT':3,
	'SIDE_LEFT':4,
	'SIDE_RIGHT':5,
}

level_dict = {
	'UNKNOWN': 0,
	'LEVEL_1': 1,
	'LEVEL_2': 2,
}

RAW_PATH = '/mnt/share_data/waymo_dataset/waymo_format/'
DATASET = 'validation'

def convert2Frame(data):
	frame = open_dataset.Frame()
	frame.ParseFromString(bytearray(data.numpy()))
	return frame

def _create_gt(obj_id, bbox, meta_data, category_name, frame_name, macro_ts, num_lidar_points_in_box):
	o = metrics_pb2.Object()
	o.context_name = frame_name
	o.score = 1.0
	o.frame_timestamp_micros = macro_ts
	# This is only needed for 2D detection or tracking tasks.
	# Set it to the camera name the prediction is for.
	# o.camera_name = camera_dict[camera_name]

	# Populating box and score.
	box = label_pb2.Label.Box()
	box.center_x = bbox.center_x
	box.center_y = bbox.center_y
	box.center_z = bbox.center_z
	box.length = bbox.length
	box.width = bbox.width
	box.height = bbox.height
	box.heading = bbox.heading
	o.object.box.CopyFrom(box)

	metadata = label_pb2.Label.Metadata()
	metadata.speed_x = meta_data.speed_x
	metadata.speed_y = meta_data.speed_y
	metadata.accel_x = meta_data.accel_x
	metadata.accel_y = meta_data.accel_y
	o.object.metadata.CopyFrom(metadata)

	# o.object.detection_difficulty_level = difficulty_level
	o.object.num_lidar_points_in_box = num_lidar_points_in_box
	# For tracking, this must be set and it must be unique for each tracked
	# sequence.
	o.object.id = obj_id
	# Use correct type.
	o.object.type = type_dict[category_name]
	return o


##############################
def cart_to_homo(mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret

############################### filt objs beyond the fov
def get3dData(segname, data, frame_id):
	# get frame
	frame = convert2Frame(data)
	frame_name = frame.context.name
	weather = frame.context.stats.weather
	time_of_day = frame.context.stats.time_of_day

	macro_ts = frame.timestamp_micros

	object_list = []

	###################################################
	T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
	camera_calibs = []
	R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]
	Tr_velo_to_cams = []
	calib_context = ''

	for camera in frame.context.camera_calibrations:
		# extrinsic parameters
		T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
			4, 4)
		T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
		Tr_velo_to_cam = \
			cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
		if camera.name == 1:  # FRONT = 1, see dataset.proto for details
			T_velo_to_front_cam = Tr_velo_to_cam.copy()
		Tr_velo_to_cam = Tr_velo_to_cam[:3, :].reshape((12, ))
		Tr_velo_to_cams.append([f'{i:e}' for i in Tr_velo_to_cam])

		# intrinsic parameters
		camera_calib = np.zeros((3, 4))
		camera_calib[0, 0] = camera.intrinsic[0]
		camera_calib[1, 1] = camera.intrinsic[1]
		camera_calib[0, 2] = camera.intrinsic[2]
		camera_calib[1, 2] = camera.intrinsic[3]
		camera_calib[2, 2] = 1
		camera_calib = list(camera_calib.reshape(12))
		camera_calib = [f'{i:e}' for i in camera_calib]
		camera_calibs.append(camera_calib)

	# all camera ids are saved as id-1 in the result because
	# camera 0 is unknown in the proto
	for i in range(5):
		calib_context += 'P' + str(i) + ': ' + \
			' '.join(camera_calibs[i]) + '\n'
	calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'
	for i in range(5):
		calib_context += 'Tr_velo_to_cam_' + str(i) + ': ' + \
			' '.join(Tr_velo_to_cams[i]) + '\n'
	
	###################################################

	R0_rect = [f'{i:e}' for i in np.eye(4).flatten()]
	R0_rect = np.array(R0_rect, dtype='float').reshape(4, 4)

	Tr = copy.deepcopy(Tr_velo_to_cams[0])
	Tr.append(0.000000e+00)
	Tr.append(0.000000e+00)
	Tr.append(0.000000e+00)
	Tr.append(1.000000e+00)
	Tr = np.array(Tr, dtype='float').reshape(4, 4)

	calib = copy.deepcopy(camera_calibs[0])
	calib.append(0.000000e+00)
	calib.append(0.000000e+00)
	calib.append(0.000000e+00)
	calib.append(1.000000e+00)
	calib = np.array(calib, dtype='float').reshape(4, 4)

	for index, laser_labels in enumerate(frame.laser_labels):
		box = laser_labels.box
		meta_data = laser_labels.metadata
		box_class_index = laser_labels.type
		box_class_name = open_dataset_label.Label.Type.Name(box_class_index)
		num_lidar_points_in_box = laser_labels.num_lidar_points_in_box
		obj_id = laser_labels.id
		if num_lidar_points_in_box == 0:
			continue
		
		bbox_np = np.array([box.center_x, box.center_y, box.center_z], dtype='float')
		bbox_np = np.expand_dims(bbox_np, axis=0)

	
		bbox_filted = box_np_ops.remove_outside_points(bbox_np, R0_rect, Tr, calib, [1280, 1920])
		
		if bbox_filted.shape[0] == 0:
			continue
		
		o = _create_gt(obj_id, box, meta_data, box_class_name, frame_name, macro_ts, num_lidar_points_in_box)
		object_list.append(o)
	
	return object_list, frame_name + '_' + str(frame_id), macro_ts

def _create_pd_file_example(obj_list):
	"""Creates a prediction objects file."""
	objects = metrics_pb2.Objects()
	for obj in obj_list:
		objects.objects.append(obj)
	# Add more objects. Note that a reasonable detector should limit its maximum
	# number of boxes predicted per frame. A reasonable value is around 400. A
	# huge number of boxes can slow down metrics computation.

	# Write objects to a file.
	f = open('/mnt/share_data/waymo_dataset/waymo_format/gt_fov.bin', 'wb')
	f.write(objects.SerializeToString())
	f.close()

if __name__ == '__main__':
	segments = glob.glob(os.path.join(RAW_PATH, DATASET, '*.tfrecord'))

	data_queue = Queue(2)
	result_queue = Queue()
	workers = []
	num_workers = 1
	
	t1 = time.time()

	object_list = []
	for i, segment in enumerate(segments):
		start_time = time.time()
		if i == 0:
			def eval_worker(data_queue, result_queue):
				while True:
					index, name, data = data_queue.get()
					roidb = get3dData(name, data, index)
					result_queue.put(roidb)
			for _ in range(num_workers):
				workers.append(Thread(target=eval_worker, args=(data_queue, result_queue)))
			for w in workers:
				w.daemon = True
				w.start()
		segname = segment.split('/')[-1].split('.')[0]
		dataset = tf.data.TFRecordDataset(segment, compression_type='')
		dataset = list(dataset)
		def data_enqueue(dataset, data_queue):
			for i, data in enumerate(dataset):
				data_queue.put((i, segname, data))
		enqueue_worker = Thread(target=data_enqueue, args=(dataset, data_queue))
		enqueue_worker.daemon = True
		enqueue_worker.start()

		for _ in range(len(dataset)):
			all_data = result_queue.get()
			obj_data, frame_id, ts = all_data
			object_list += obj_data
		end_time = time.time()
		print("Finished Processing Segment %d/%d in %g s"%(i, len(segments), end_time - start_time))
	t2 = time.time()
	print("Parsing Done in %g s" % (t2 - t1))

	_create_pd_file_example(object_list)

	t3 = time.time()
	print("Saving GT File in %g s" % (t3 - t2))