# Synthetic Cookbook for Using/Testing/Demonstrating VisualDet3D in ROS

One of the most often required features of visual detection 3D is a full tool chain including data / model training / model testing / ROS deployment / demonstration and more.

In this cookbook, a set of tools are introduced to help this entire process.

It mainly involves three open-source repo:

- [kitti_visualize]
- [visualDet3D]
- [visualDet3D_ros]
  
Menu:

<!-- vscode-markdown-toc -->
- [Synthetic Cookbook for Using/Testing/Demonstrating VisualDet3D in ROS](#synthetic-cookbook-for-usingtestingdemonstrating-visualdet3d-in-ros)
  - [1. Data](#1-data)
  - [2. Training / Numerical Testing](#2-training--numerical-testing)
  - [3. Visualize offline test results (Optional)](#3-visualize-offline-test-results-optional)
  - [4. Visualize real-time prediction results in ROS.](#4-visualize-real-time-prediction-results-in-ros)
  - [5. Visualize real-time streaming in ROS.](#5-visualize-real-time-streaming-in-ros)
  - [6. Streaming results on other/customized datasets](#6-streaming-results-on-othercustomized-datasets)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


##  1. <a name='Data'></a>Data

Download the [KITTI object dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) to local. The file structure should end up like /data/kitti/{training/testing}/{image_2/calib/label_2...}.

Please follow the setup process of [kitti_visualize] and make sure you could correctly visualize kitti object data (**unselect** isSequential in the GUI, and playing around with the index number). 

Now you can play around with the data and also confirm the file structure is fine.

##  2. <a name='TrainingNumericalTesting'></a>Training / Numerical Testing

Checkout the training pipeline in [visualDet3D for mono3D](https://github.com/Owen-Liuyuxuan/visualDet3D/blob/master/docs/mono3d.md).

If you do not want to train, one of the methods is to download the **code and assets** at [Release 1.0 of visualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D/releases/tag/1.0) to reproduce the result of a pre-uploaded checkpoint. 

##  3. <a name='VisualizeofflinetestresultsOptional'></a>Visualize offline test results (Optional)

Checkout the [additional labels](https://github.com/Owen-Liuyuxuan/kitti_visualize#additional-labels) part. 

Basically, you need to copy the texts under *workdirs/Yolo3D/output/testing/data* of visualDet3D into  */data/kitti_obj/testing/additional_label_2*, and launch [kitti_visualize] on the testing split to visual the offline results.

##  4. <a name='Visualizereal-timepredictionresultsinROS.'></a>Visualize real-time prediction results in ROS.

Checkout the setup process of [visualDet3D_ros]. Notice to modify the "visual3d_path", "cfg_file", "weight_path" parameters based on your own setting of visualDet3D.

Launch both [kitti_visualize] and [visualDet3D_ros], **unselect** isSequential in the GUI. With the default launch file, the two node publish and read on the same image/camera_param topics. By adjusting what is shown in the RVIZ, we can now visualize the inference result in real-time. 


##  5. <a name='Visualizereal-timestreaminginROS.'></a>Visualize real-time streaming in ROS.

We suggest download the KITTI [raw dataset](https://www.cvlibs.net/datasets/kitti/raw_data.php) to streaming out image/lidar data.

1. Please follow the setup process of [kitti_visualize] and make sure you could correctly visualize kitti raw data (**select** isSequential in the GUI, and playing around with the index number). 

2. Launch visualDet3D_ros as usual.

With the sequence data in [kitti_visualize] keep streaming into ROS, visualDet3D_ros will also streamingly conduct inference on the images, producing demo results like the one on the readme page of [visualDet3D_ros]. 

Notice, that the IO/computational stress will be high at this step. You could adjust the streaming speed of [kitti_visualize] by changing local param in the launch file(UPDATE_FREQUENCY) or directly modifying the code.

##  6. <a name='Streamingresultsonothercustomizeddatasets'></a>Streaming results on other/customized datasets

After sucessfully setting up [visualDet3D_ros], practically we can test the inference of [visualDet3D] on any image streams with ROS interface (like USB webcam).

We provide ROS interface to [nuScenes dataset](https://github.com/Owen-Liuyuxuan/nuscenes_visualize) and [KITTI-360 dataset](https://github.com/Owen-Liuyuxuan/kitti360_visualize). 

To get more robust results, training on customized datasets, or trying camera-insensitive algorithms (like the adapted version of MonoFlex in visualDet3D) worth the efforts.

By adapting your own dataset to kitti/kitti360/nuscenes formats, you can also make use of these existing tools to boost development of your 3D detection project.


[kitti_visualize]:https://github.com/Owen-Liuyuxuan/kitti_visualize
[visualDet3D]:https://github.com/Owen-Liuyuxuan/visualDet3D
[visualDet3D_ros]:https://github.com/Owen-Liuyuxuan/visualDet3D_ros