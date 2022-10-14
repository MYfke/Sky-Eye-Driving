"""
此代码实现了多相机图像的展示
"""

from camera.MyCamera import *
from train import *


def demo():
    # 初始化相机，返回数据流对象的列表
    cameraCnt, cameraList, streamSourceList, userInfoList = init_camera()
    # TODO 此处开始建立模型，然后在回调函数中向模型传入图片
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

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
