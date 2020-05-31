import cv2
import numpy as np
import math


class detect(object):
    def __init__(self, image):
        self.image = image
        blurred = cv2.GaussianBlur(image, (3, 3), 0)  # 高斯模糊，降噪
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)  # 转为灰度图片
        autosobel = cv2.Canny(gray, 150, 200, 3)  # sobel变化，边缘检测
        contours, _ = cv2.findContours(autosobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detect_cont = self.filter_contours(contours)
        self.classified_cont = self.classify_cont(detect_cont)

    def filter_contours(self, con):  # 过滤得到疑似塑件的轮廓
        filtered = []
        square = [cv2.contourArea(x) for x in con]
        for i in range(len(square)):
            if 3500 < square[i] < 7500:
                filtered.append(con[i])
        return filtered

    def classify_cont(self, cont):  # 分类识别出的轮廓
        model_cont = np.load("./contdata.npy", allow_pickle=True)
        clsy_rlt = []  # 存放分类结果，格式[[轮廓，类型，区别度]...]
        for i, k in enumerate(model_cont):
            for j in cont:
                m = cv2.matchShapes(k, j, 1, 0.0)
                if m < 0.035:
                    clsy_rlt.append([j, i, m])
        clsy_rlt.sort(key=lambda x: x[2])
        return clsy_rlt  # 使用区分度排序

    def get_classified_pic(self):
        pic = self.image.copy()
        for i, c in enumerate(self.classified_cont):
            if i < 3:  # 前三个标绿，后面标红
                cv2.drawContours(pic, [c[0]], -1, (0, 255, 0), 5)
            else:
                cv2.drawContours(pic, [c[0]], -1, (0, 0, 255), 5)
        return True, pic  # 返回识别成功，识别后的图像

    def get_three_cont(self):
        return self.classified_cont[0:3]  # 返回排序后的前三个轮廓

    def get_pic_info(self, cont):
        pics = []
        infos = []
        for i in cont:
            rect = cv2.minAreaRect(i[0])
            box = cv2.boxPoints(rect)  # 获得四个端点坐标
            box = np.int0(box)  # 变化为整型
            # img = cv2.drawContours(self.image.copy(), [box], -1, (0, 255, 0), 5)
            # 找出中点，标出中心点
            center_p = (int(rect[0][0]), int(rect[0][1]))
            # 计算出长宽和倾斜角
            dst_w = int(math.floor(rect[1][0] / 2))
            dst_h = int(math.floor(rect[1][1] / 2))
            angle = rect[2]
            if dst_h > dst_w:
                dst_w, dst_h = dst_h, dst_w
                angle += 90
            M = cv2.getRotationMatrix2D(center_p, angle, 1.0)  # 获取旋转参数

            rotated = cv2.warpAffine(self.image, M, (self.image.shape[0:2]))
            crop = rotated[center_p[1] - dst_h - 2:center_p[1] + dst_h + 2,
                   center_p[0] - dst_w - 2:center_p[0] + dst_w + 2]
            # 区分shape0
            if i[1] == 0:
                # 在白底上绘制轮廓
                w_bg = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
                w_bg.fill(0)
                cv2.drawContours(w_bg, [i[0]], -1, 255, 2)
                # 旋转使轮廓水平
                rotated_wbg = cv2.warpAffine(w_bg, M, (self.image.shape[0:2]))
                crop_wbg = rotated_wbg[center_p[1] - dst_h - 5:center_p[1] + dst_h + 5,
                           center_p[0] - dst_w - 5:center_p[0] + dst_w + 5]
                # 检测轮廓，得到轮廓list（只有一个）
                c, _ = cv2.findContours(crop_wbg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # sss = cv2.cvtColor(crop_wbg.copy(), cv2.COLOR_GRAY2BGR)
                # sss = cv2.drawContours(sss, [c[0]], -1, (0,255,0), 1)
                # cv2.imshow("www", sss)
                # 缺陷检测，检测缺口的位置，即距离凸包最远的点
                hull = cv2.convexHull(c[0], returnPoints=False)
                defects = cv2.convexityDefects(c[0], hull)
                defects_list = defects.tolist()
                defects_list.sort(key=lambda x: x[0][3], reverse=True)
                s, e, f, d = defects_list[0][0]  # 得到的list中每项是[[x,y,z,k]]格式，所以需要0
                start = tuple(c[0][s][0])
                end = tuple(c[0][e][0])
                far = tuple(c[0][f][0])
                if (far[0] > 50 & far[1] > 30) | (far[0] < 50 & far[1] < 30):
                    i[1] = 3
                # print(far)
                # distant =
                # cv2.line(crop_wbg, start, end, [0, 255, 0], 2)
                # cv2.circle(crop_wbg, far, 5, [0, 0, 255], -1)
                # cv2.imshow(f'img{i[2]}', crop_wbg)
                # i[1] = self.classify_shape0(i)
            infos.append([center_p, angle, i[1], i[2]])  # [中心点坐标，倾斜角度，类型，区别度]
            pics.append(crop)
        return pics, infos
