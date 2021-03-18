time: 20210318
pdf_source: https://www.ram-lab.com/papers/2021/yuxuan2021mono3d.pdf
code_source: https://github.com/Owen-Liuyuxuan/visualDet3D
short_title:  YOLOStereo3D: A Step Back to 2D for Efficient Stereo 3D Detection

#  YOLOStereo3D: A Step Back to 2D for Efficient Stereo 3D Detection (EN)
This is my paper accepcted by *ICRA* 2021. The open-sourced code is in https://github.com/Owen-Liuyuxuan/visualDet3D .

The basic idea is to train Stereo 3D detection model "like" a Monocular one, to obtain fast inference speed and reasonable performance. Multiple modules are introduced and merged.

The re-production of the stereo/monocular results of this paper should be rather stable provided with the open-source repo.

## Core Operations and Code Placement

- Precomputing statistics for anchors: [script github page](https://github.com/Owen-Liuyuxuan/visualDet3D/blob/master/scripts/imdb_precompute_3d.py)
- Using the statistics for anchors: [head github page](https://github.com/Owen-Liuyuxuan/visualDet3D/blob/master/visualDet3D/networks/heads/detection_3d_head.py)
- Matching Module: [lib github page](https://github.com/Owen-Liuyuxuan/visualDet3D/blob/master/visualDet3D/networks/lib/PSM_cost_volume.py)
- Ghost Module: [lib github page](https://github.com/Owen-Liuyuxuan/visualDet3D/blob/master/visualDet3D/networks/lib/ghost_module.py)
- Multi-Scale Fusion: [core github page](https://github.com/Owen-Liuyuxuan/visualDet3D/blob/master/visualDet3D/networks/detectors/yolostereo3d_core.py)
- Disparity Loss: [loss github page](https://github.com/Owen-Liuyuxuan/visualDet3D/tree/master/visualDet3D/networks/lib/disparity_loss)
- To obtain the monocular results in this paper, just trained the with the monocular 3D settings like the [GAC](GroundAwareConvultion.md) paper and get rid of additiona modules (DeformConv/GAC/...).

## Result for the published model:

[Release Page](https://github.com/Owen-Liuyuxuan/visualDet3D/releases/tag/1.1)

| Benchmark                  |  Easy   | Moderate |  Hard   |
| -------------------------- | :-----: | :------: | :-----: |
| Car Detection              | 94.75 % | 84.50 %  | 62.13 % |
| Car Orientation            | 93.65 % | 82.88 %  | 60.92 % |
| Car 3D Detection           | 65.77 % | 40.71 %  | 29.99 % |
| Car Bird's Eye View        | 74.00 % | 49.54 %  | 36.30 % |
| Pedestrian Detection       | 58.34 % | 49.54 %  | 36.30 % |
| Pedestrian Orientation     | 50.41 % | 36.81 %  | 31.51 % |
| Pedestrian 3D Detection    | 31.03 % | 20.67 %  | 18.34 % |
| Pedestrian Bird's Eye View | 32.52 % | 22.74 %  | 19.16 % |
