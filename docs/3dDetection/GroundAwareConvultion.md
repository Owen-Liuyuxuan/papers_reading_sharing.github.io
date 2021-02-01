time: 20210201
pdf_source: https://www.ram-lab.com/papers/2021/yuxuan2021mono3d.pdf
code_source: https://github.com/Owen-Liuyuxuan/visualDet3D
short_title: Ground-aware Monocular 3D Object Detection for Autonomous Driving

# Ground-aware Monocular 3D Object Detection for Autonomous Driving (EN)
This is my paper accepcted by *RAL* 2021. The open-sourced code is in https://github.com/Owen-Liuyuxuan/visualDet3D .

The basic idea is an attempt to mimic how people perceive depth from a single image, and more importantly try to incorperate calibration matrix and ground plane information into the detection model.

## Core Operations and Code Placement

- Precomputing statistics for anchors: [script github page](https://github.com/Owen-Liuyuxuan/visualDet3D/blob/master/scripts/imdb_precompute_3d.py)
- Using the statistics for anchors: [head github page](https://github.com/Owen-Liuyuxuan/visualDet3D/blob/master/visualDet3D/networks/heads/detection_3d_head.py)
- Ground-Aware Convolution Module: [block github page](https://github.com/Owen-Liuyuxuan/visualDet3D/blob/master/visualDet3D/networks/lib/look_ground.py)
- Change the "cfg.detector.name" in config to **Yolo3D** and experiment with DeformConv (which also provide robust and top performance). 

## Result for the published model:

[Release Page](https://github.com/Owen-Liuyuxuan/visualDet3D/releases/tag/1.0)

| Benchmark             | Easy |   Moderate  |   Hard   |
|---------------------|:--------:|:-------:|:-------:|
| Car Detection             |  92.35  | 79.57 | 59.61 | 
| Car Orientation             | 90.87  |77.47 | 57.99 | 
| Car 3D Detection             |  21.60 | 13.17  | 9.94  | 
| Car Bird's Eye View             |  29.38 | 18.00 | 13.14 | 
