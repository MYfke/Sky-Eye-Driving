"""
此代码实现了棋盘格标定，标定图片目录结构如下
${IMG_DIR}
    |--C0_C1
    |   |--C0
    |   |   |--01,jpg
    |   |   |--02,jpg
    |   |   |--03,jpg
    |   |    '''
    |   |
    |   |--C1
    |       |--01,jpg
    |       |--02,jpg
    |       |--03,jpg
    |       '''
    |--C1_C2
    '''

"""

import cv2
import numpy as np
import os
import yaml

IMG_DIR = '../data/calibration_image'  # 标定图片路径
IMG_SIZE = (640, 480)  # 标定图片路尺寸
CORNER_ROW = 7  # 每行标定板内点个数
CORNER_COLUMN = 7  # 每列标定板内点个数
GRID_SIZE = 10  # 标定板方格尺寸


def getImageDict(img_dir):
    # 获取图片文件夹位置，方便opencv读取
    # 参数：照片文件路径
    # 返回值：数组，每一个元素表示一张照片的绝对路径
    imgPath = {}
    for dir in os.listdir(img_dir):
        files = []
        for d in os.listdir(os.path.join(img_dir, dir)):
            files.append([os.path.join(img_dir, dir, d, f) for f in os.listdir(os.path.join(img_dir, dir, d, ))])
        imgPath[dir] = files

    return imgPath


def getObjectPoints(m, n, k):
    # 计算真实坐标
    # 参数：内点行数，内点列， 标定板大小
    # 返回值：数组，（m*n行，3列），真实内点坐标
    objP = np.zeros(shape=(m * n, 3), dtype=np.float32)
    for i in range(m * n):
        objP[i][0] = i % m
        objP[i][1] = int(i / m)
    return objP * k


if __name__ == '__main__':
    # 相机标定参数设定（单目，双目）
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 计算标定板真实坐标，标定板内点，大小10mm*10mm
    objPoint = getObjectPoints(CORNER_ROW, CORNER_COLUMN, 10)

    objPoints = [[], []]
    imgPoints = [[], []]
    rt_matrix = {}

    filePath = getImageDict(IMG_DIR)

    for key, imgPath in filePath.items():
        for i in range(0, len(imgPath)):
            for path in imgPath[i]:
                # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
                image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
                # 分别读取每张图片并转化为灰度图
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # opencv寻找角点
                ret, corners = cv2.findChessboardCorners(gray, (CORNER_ROW, CORNER_COLUMN), None)
                # opencv对真实坐标格式要求，vector<vector<Point3f>>类型
                objPoints[i].append(objPoint)
                # 角点细化
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                imgPoints[i].append(corners2)

        # 对左右相机分别进行单目相机标定（复制时格式可能有点问题，用pycharm自动格式化）
        ret1, cameraMatrix1, distMatrix1, R1, T1 = cv2.calibrateCamera(objPoints[0], imgPoints[0], IMG_SIZE, None, None)
        ret2, cameraMatrix2, distMatrix2, R2, T2 = cv2.calibrateCamera(objPoints[1], imgPoints[1], IMG_SIZE, None, None)
        # 双目相机校正
        retS, mLS, dLS, mRS, dRS, R, T, E, F = cv2.stereoCalibrate(objPoints[0], imgPoints[0], imgPoints[1],
                                                                   cameraMatrix1, distMatrix1,
                                                                   cameraMatrix2, distMatrix2,
                                                                   IMG_SIZE, criteria_stereo,
                                                                   flags=cv2.CALIB_USE_INTRINSIC_GUESS)

        rt_matrix[key] = {"retS": retS, 'mLS': mLS, 'dLS': dLS, 'mRS': mRS, 'dRS': dRS, 'R': R, 'T': T, 'E': E, 'F': F}
        # 标定结束，结果输出，cameraMatrixL，cameraMatrixR分别为左右相机内参数矩阵
        # R， T为相机2与相机1旋转平移矩阵
        print(f'{key}两相机标定完成')
        print(f'旋转矩阵R:\n{R}')
        print(f'平移矩阵T:\n{T}')
        print('-' * 50)

    print('所有相机标定完成')
    with open('../data/RT.yaml', 'w') as f:
        yaml.dump(rt_matrix, f)
