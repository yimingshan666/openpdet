import copy
import pickle

import numpy as np
from skimage import io

from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate


## 定义kitti数据集的类
class KittiDataset(DatasetTemplate):
    ## 不用改
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """

        ## 初始化，继承父类
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

##### 我觉得，这上面六十多行，都是一些初始化，以及读取文件路径，识别是train还是test还是val的东西，不需要做改动#####


## 读取bin文件，雷达信息，四列数据的那个，不用改  get_lidar
    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        ## 数据转换为numpy而且是四列，xyz intensity
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

##读取图片信息，直接删除  get_image / get_image_shape
    # def get_image(self, idx):
    #     """
    #     Loads image for a sample
    #     Args:
    #         idx: int, Sample index
    #     Returns:
    #         image: (H, W, 3), RGB Image
    #     """
    #     img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
    #     assert img_file.exists()
    #     image = io.imread(img_file)
    #     image = image.astype(np.float32)
    #     image /= 255.0
    #     return image
    #
    # def get_image_shape(self, idx):
    #     img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
    #     assert img_file.exists()
    #     return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

## 读取lables 不用删除  get_label
    ###根据序列号，获取标签信息
    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

## 不知道这块是干嘛的？？？？ 文件后缀有.png 可能和图片相关，先删除掉 get_depth_map
    # def get_depth_map(self, idx):
    #     """
    #     Loads depth map for a sample
    #     Args:
    #         idx: str, Sample index
    #     Returns:
    #         depth: (H, W), Depth map
    #     """
    #     depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
    #     assert depth_file.exists()
    #     depth = io.imread(depth_file)
    #     depth = depth.astype(np.float32)
    #     depth /= 256.0
    #     return depth


## 做calib的，直接删除  get_calib
    # def get_calib(self, idx):
    #     calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
    #     assert calib_file.exists()
    #     return calibration_kitti.Calibration(calib_file)

## optional的，用不用到这部分，都不需要删除  get_road_plane


    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane


### 这个还不确定要不要改，（暂时觉得可以全部删除掉）虽然pts_rect看起来有用，实际上是为calib服务的，可以删掉
    ## 调用矫正类中的方法，将点的直角坐标转为相机坐标
    # @staticmethod  ##静态方法范围？？？？  get_fov_flag
    # def get_fov_flag(pts_rect, img_shape, calib):
    #     """
    #     Args:
    #         pts_rect:
    #         img_shape:
    #         calib:
    #
    #     Returns:
    #
    #     """
    #     pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    #     val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    #     val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    #     val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    #     pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    #
    #     return pts_valid_flag


#### 获取信息 有些不用的要改掉####暂时还没明白要咋改？？ 暂时改好了
    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        ## 处理单帧数据
        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {} #空字典
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info  #目前的信息加入info字典

            ## 处理image信息，删除
            # image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            # info['image'] = image_info
            # calib = self.get_calib(sample_idx)

            ## 一些相机坐标的，删除
            # P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            # R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            # R0_4x4[3, 3] = 1.
            # R0_4x4[:3, :3] = calib.R0
            # V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            # calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            #
            # info['calib'] = calib_info

            if has_label:
                # 调用get_label中的get_objects_from_label函数，首先读取该文件的所有行 赋值为 lines
                # 在对lines中的每一个line（一个object的参数）作为object3d类的参数 进行遍历，
                # 最后返回：objects[]列表 ,里面是当前文件里所有物体的属性值，如：type、x,y,等
                obj_list = self.get_label(sample_idx)

                # 定义一个空字典，annotations是注解的意思，obj_list读入的东西，放进annotations里
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])   # 物体名字
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list]) # 物体是否被截断
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])  #物体是否被遮挡
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])  #物体观察角度
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)  ## 物体的二维边界，xmin，ymin，xmax，ymax
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format  ## 3维物体的尺寸 长宽高 单位m
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0) ## 3维物体的位置 xyz 单位m
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list]) ## 3维物体的空间方向：rotation_y  ##kitti的label中，数据到这里就结束了，后俩没有
                annotations['score'] = np.array([obj.score for obj in obj_list])  ##检测的置信度
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32) ## 这是啥？？

                ## 计算有效物体的个数，如10个，object除去“DontCare”4个，还剩num_objects 6个
                ## 自己的label里没有Dontcare，暂时留下，不知道会不会影响？？？？？
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                ## 物体总个数
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                # 假设有效物体的个数是N
                # 取有效物体的 location（N,3）、dimensions（N,3）、rotation_y（N,1）信息
                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                ## 有一些和calib相关的，通过计算得到在lidar坐标系下的坐标,应该是需要修改，不能直接删除
                ## 直接把带有calib计算的，改为读进来的location就是雷达坐标系下坐标。

                # 读进来的location就是雷达坐标系下的
                loc_lidar = loc
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                # loc_lidar[:, 2] += h[:, 0] / 2
                # 下面计算得到的gt_boxes_lidar是(N,7) np.concatenate是拼接的意思
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                ### 这一块是想得到到底有多少个点，需要修改，不能直接删除
                ## 修改为 直接从lindar中读进几个就是几个点
                ### 等会儿试一试，这一块删除掉后，会不会影响结果。（计算有几个点）
                # if count_inside_pts:
                #     points = self.get_lidar(sample_idx)
                    # calib = self.get_calib(sample_idx)
                    # pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    #
                    # fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    # pts_fov = points[fov_flag]
                    # corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    # ### 有效物体的个数
                    # num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    #
                    # # in_hull函数是判断点云是否在bbox中，（是否在物体的2D检测框中），我觉得可以删除掉
                    # for k in range(num_objects):
                    #     flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                    #     num_points_in_gt[k] = flag.sum()
                    # annotations['num_points_in_gt'] = points

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)


##### 一些真实的信息，我觉得大的不用改动，有bbox需要改，暂时先不改bbox，
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        # 调取infos里的每个info的信息，一个info是一帧的数据
        for k in range(len(infos)):
            # 输出的是 第几个样本 如7/780
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            # 取当前帧的信息 info
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']

            ##获取.bin中点的信息
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']
            # num_obj是有效物体的个数，为N
            num_obj = gt_boxes.shape[0]

            # 维度是（N,M）,  N是有效物体的个数，M是点云的个数，再转化为numpy
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)



### 有一些calib和image相关的信息要改,还修改了几个box_utils.py里的代码
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        ## 获取预测后的模版字典，先全置为0
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        ## 生成一个帧的预测字典，预测后的数据放入之前的模版里
        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()

            ##义一个帧的空字典，用来存放来自预测的信息
            pred_dict = get_template_prediction(pred_scores.shape[0])

            ##如果没有物体，则返回空字典
            if pred_scores.shape[0] == 0:
                return pred_dict

            ## 雷达/相机坐标系转换相关的，需要改！！！！！！！ 暂时还不知道咋改,还有bbox和image相关的，要改
            # calib = batch_dict['calib'][batch_index]
            # image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            ###这一步需要的，把表示3d_boxes的方法转换一下
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes)
            ## 通过3d_boxes 计算出2d_boxes边界，计算iou的时候会用到
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera
            )




            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            len1 = len(pred_boxes)
            ##经过核查，是bbox=0，导致的ap=0，在这里修改下
            # pred_dict['bbox'] = np.zeros((len1,4)) ## 这里bbox是随便自己写的，其实没用
            pred_dict['bbox'] = pred_boxes_img
            # pred_dict['bbox'] =
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        ## 预测信息，需要稍微处理下，然后输出
        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

#### 我觉得不用改
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)  ##预测的
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos] ##标注的
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        # 额外的
        # print(eval_det_annos[0])  ## 额外加上的，测试一下
        # print(eval_gt_annos)
        return ap_result_str, ap_dict

#### 我觉得不用改
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)


####加载数据，并做一些处理，需要改，我觉得改完了
    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        ## 一些image要改 calib也要改
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        # img_shape = info['image']['image_shape']
        # calib = self.get_calib(sample_idx)
        # get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
        }

        ## 这里也要改，一些我的label里没有的和坐标系转换的
        if 'annos' in info:
            ## 把这一帧中的annos弄出来
            annos = info['annos']
            ##在info中剔除包含'DontCare'的数据信息
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            # 得到有效物体object(N个)的位置、大小和角度信息（N,3）,(N,3),(N)
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            ## 把boxes拼接成7列
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            ## 把表示boxes的方式变一下
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

            # if "gt_boxes2d" in get_item_list:
            #     input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        # if "points" in get_item_list:
        #     points = self.get_lidar(sample_idx)
        #     # if self.dataset_cfg.FOV_POINTS_ONLY:
        #         # pts_rect = calib.lidar_to_rect(points[:, 0:3])
        #         # fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
        #         # points = points[fov_flag]
        #     input_dict['points'] = points

        # if "images" in get_item_list:
        #     input_dict['images'] = self.get_image(sample_idx)

        # if "depth_maps" in get_item_list:
        #     input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        # if "calib_matricies" in get_item_list:
        #     input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        # input_dict['calib'] = calib
        data_dict = self.prepare_data(data_dict=input_dict)

        # data_dict['image_shape'] = img_shape
        return data_dict


#### 各种参数传递，保存文件，等等 暂时觉得应该不需要改，但是还要细看下到底用不用改。
def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


### 我觉得不用动
if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )
