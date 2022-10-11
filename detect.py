import time
from camera.MyCamera import *
from train import *


def demo():
    # 初始化相机，返回数据流对象的列表
    cameraCnt, cameraList, streamSourceList, userInfoList = init_camera()
    # TODO 此处开始建立模型，然后在回调函数中向模型传入图片
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    '''
    is_distributed = init_distributed(args)
    master = True
    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0

    if is_distributed:
        device = torch.device(args.local_rank)
    else:
        device = torch.device('cpu')  # TODO 这里可以更改设备

    # config  配置
    config = cfg.load_config(args.config)
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    # 有三种模型，此处开始创建模型
    model = {
        # "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        # "vol": VolumetricTriangulationNet
    }[config.model.name](config, device=device).to(device)

    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint, map_location='cpu')
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pretrained weights for whole model")

    # criterion  损失函数
    criterion_class = {
        "MSE": KeypointsMSELoss,
        "MSESmooth": KeypointsMSESmoothLoss,
        "MAE": KeypointsMAELoss
    }[config.opt.criterion]

    if config.opt.criterion == "MSESmooth":
        criterion = criterion_class(config.opt.mse_smooth_threshold)
    else:
        criterion = criterion_class()

    # optimizer  优化器
    opt = None
    if not args.eval:
        if config.model.name == "vol":
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(),
                  'lr': config.opt.process_features_lr if hasattr(config.opt,
                                                                  "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(),
                  'lr': config.opt.volume_net_lr if hasattr(config.opt, "volume_net_lr") else config.opt.lr}
                 ],
                lr=config.opt.lr
            )
        else:
            opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt.lr)

    # datasets  数据集
    print("Loading data...")
    train_dataloader, val_dataloader = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment  实验
    experiment_dir, writer = None, None
    if master:
        experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)

    if not args.eval:
        # train loop  训练循环
        n_iters_total_train, n_iters_total_val = 0, 0
        for epoch in range(config.opt.n_epochs):

            n_iters_total_train = one_epoch(model, criterion, opt, config, train_dataloader, device, epoch,
                                            n_iters_total=n_iters_total_train, is_train=True, master=master,
                                            experiment_dir=experiment_dir, writer=writer)
            n_iters_total_val = one_epoch(model, criterion, opt, config, val_dataloader, device, epoch,
                                          n_iters_total=n_iters_total_val, is_train=False, master=master,
                                          experiment_dir=experiment_dir, writer=writer)

            if master:
                checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)

                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))

            print(f"{n_iters_total_train} iters done.{n_iters_total_train} 迭代完成。")
    else:
        if args.eval_dataset == 'train':
            one_epoch(model, criterion, opt, config, train_dataloader, device, 0, n_iters_total=0, is_train=False,
                      master=master, experiment_dir=experiment_dir, writer=writer)
        else:
            one_epoch(model, criterion, opt, config, val_dataloader, device, 0, n_iters_total=0, is_train=False,
                      master=master, experiment_dir=experiment_dir, writer=writer)

    print("Done.")
    '''

    for index in range(0, cameraCnt):
        # 开始拉流
        # start grabbing
        nRet = streamSourceList[index].contents.startGrabbing(streamSourceList[index], c_ulonglong(0),
                                                              c_int(GENICAM_EGrabStrategy.grabStrartegySequential))
        if nRet != 0:
            print("startGrabbing fail!")
            # 释放相关资源
            # release stream source object before return
            streamSourceList[index].contents.release(streamSourceList[index])
            return -1

    # while True:
    #     if len(package()) > 5:
    #         print(package())
    #         package().clear()
    #     if cv2.waitKey(1):
    #         global g_isStop
    #         g_isStop = 1
    #         break

        # images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(
        #     batch, device, config)
        #
        # keypoints_2d_pred, cuboids_pred, base_points_pred = None, None, None
        # if model_type == "alg" or model_type == "ransac":
        #     keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch,
        #                                                                                   proj_matricies_batch,

    # 自由拉流 x 秒
    # grabbing x seconds
    time.sleep(g_Image_Grabbing_Timer)
    global g_isStop
    g_isStop = 1

    for index in range(0, cameraCnt):
        # 反注册回调函数
        # unsubscribe grabbing callback
        nRet = streamSourceList[index].contents.detachGrabbingEx(streamSourceList[index], frameCallbackFuncEx,
                                                                 userInfoList[index])
        if nRet != 0:
            print("detachGrabbingEx fail!")
            # 释放相关资源
            # release stream source object before return
            streamSourceList[index].contents.release(streamSourceList[index])
            return -1

        # 停止拉流
        # stop grabbing
        nRet = streamSourceList[index].contents.stopGrabbing(streamSourceList[index])
        if nRet != 0:
            print("stopGrabbing fail!")
            # 释放相关资源
            # release stream source object before return
            streamSourceList[index].contents.release(streamSourceList[index])
            return -1

        # cv2.destroyAllWindows()

        # 关闭相机
        # close camera
        camera = cameraList[index]
        nRet = closeCamera(camera)
        if nRet != 0:
            print("closeCamera fail")
            # 释放相关资源
            # release stream source object before return
            streamSourceList[index].contents.release(streamSourceList[index])
            return -1

        # 释放相关资源
        # release stream source object at the end of use
        streamSourceList[index].contents.release(streamSourceList[index])

    return 0


if __name__ == "__main__":

    nRet = demo()
    if nRet != 0:
        print("Some Error happend")
    print("--------- Demo end ---------")
    # 3s exit
    time.sleep(0.2)
