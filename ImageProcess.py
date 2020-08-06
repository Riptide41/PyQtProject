import cv2
import numpy as np
import math


class Detect(object):
    def __init__(self, image, min_square, max_square):
        self.image = image
        # blurred = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯模糊，降噪

        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
        # blurred = cv2.filter2D(image, -1, kernel=kernel)
        # cv2.imshow("ruihua", blurred)
        # blurred = cv2.bilateralFilter(blurred, 0, 100, 15)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图片
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)  # 高斯模糊，降噪

        ret, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("binary pic", binary)
        # autosobel = cv2.Canny(blurred, 150, 200, 3)  # sobel变化，边缘检测
        # cv2.imshow(" pic", autosobel)

        cv2.imshow("binary pic", binary)

        self.contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.detect_min_square = min_square
        self.detect_max_square = max_square
        self.detect_cont = self.filter_contours()
        self.classified_cont = self.classify_cont()

    def filter_contours(self):  # 过滤得到疑似塑件的轮廓
        filtered = []
        square = [cv2.contourArea(x) for x in self.contours]
        for i in range(len(square)):
            if self.detect_min_square < self.contours[i].size < self.detect_max_square:
                print(self.contours[i].size)
                print("area:", cv2.contourArea(self.contours[i]))
                filtered.append(self.contours[i])

        # ******************测试使用***********************
        # test = cv2.drawContours(self.image.copy(), [filtered[16]], -1, 255, 5)
        # print("test size:", filtered[3].size)
        # model_cont = np.load("./contdata.npy", allow_pickle=True)
        # model_cont[2] = filtered[13]
        # np.save("./contdata.npy", model_cont)

        # cv2.imshow("test", test)
        return filtered

    def classify_cont(self):  # 分类识别出的轮廓
        model_cont = np.load("./contdata.npy", allow_pickle=True)
        clsy_rlt = []  # 存放分类结果，格式[[轮廓，类型，区别度]...]
        for i, k in enumerate(model_cont):
            for j in self.detect_cont:
                m = cv2.matchShapes(k, j, 1, 0.0)
                if m < 0.035:
                    clsy_rlt.append([j, i, m])
                    print("detected aread:", j.size)
        clsy_rlt.sort(key=lambda x: x[2])
        return clsy_rlt  # 使用区分度排序

    def get_classified_pic(self):
        pic = self.image.copy()
        for i, c in enumerate(self.classified_cont):
            if i < 4:  # 前三个标绿，后面标红
                cv2.drawContours(pic, [c[0]], -1, (0, 255, 0), 5)
            else:
                cv2.drawContours(pic, [c[0]], -1, (0, 0, 255), 5)
        return True, pic  # 返回识别成功，识别后的图像

    def get_four_pic_info(self):
        pics = []
        infos = []
        cont_num = 0
        for i in self.classified_cont:
            flag, pic, info = self.get_pic_info(i)
            if flag:
                infos.append(info)  # [中心点坐标，倾斜角度，类型，区别度]
                pics.append(pic)
                cont_num += 1
            if cont_num == 4:
                break
        while cont_num != 4:
            pics.append(cv2.imread("./Faildetect.png"))
            infos.append([("NaN", "NaN"), "NaN", "NaN", "NaN"])
            cont_num += 1
        print("cont_num:", cont_num)
        return True, pics, infos  # 返回排序后的前四个个轮廓

    def get_pic_info(self, cont):
        rect = cv2.minAreaRect(cont[0])
        cv2.boxPoints(rect)  # 获得四个端点坐标
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
        print("sssss", self.image.shape[0:2])
        rotated = cv2.warpAffine(self.image, M, (1623,1080))
        crop = rotated[center_p[1] - dst_h - 2:center_p[1] + dst_h + 2,
               center_p[0] - dst_w - 2:center_p[0] + dst_w + 2]
        # 当塑件处于边缘时，截取的图片size为0，抛弃
        if crop.size is 0:
            return False, crop, [center_p, angle, cont[1], cont[2]]
        # if crop.empty():
        #     print("error occur")

        # 区分shape2
        if cont[1] == 2:
            # 在白底上绘制轮廓
            w_bg = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
            w_bg.fill(0)
            cv2.drawContours(w_bg, [cont[0]], -1, 255, 2)
            # 旋转使轮廓水平
            rotated_wbg = cv2.warpAffine(w_bg, M, (self.image.shape[0:2]))
            crop_wbg = rotated_wbg[center_p[1] - dst_h - 5:center_p[1] + dst_h + 5,
                       center_p[0] - dst_w - 5:center_p[0] + dst_w + 5]
            # 检测轮廓，得到轮廓list（只有一个）
            # cv2.imshow("ss", crop_wbg)
            c, _ = cv2.findContours(crop_wbg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if c:
                # sss = cv2.cvtColor(crop_wbg.copy(), cv2.COLOR_GRAY2BGR)
                # sss = cv2.drawContours(sss, [c[0]], -1, (0,255,0), 1)
                # cv2.imshow("www", sss)
                # 缺陷检测，检测缺口的位置，即距离凸包最远的点
                hull = cv2.convexHull(c[0], returnPoints=False)
                # print(c[0])
                defects = cv2.convexityDefects(c[0], hull)
                print(defects)
                defects_list = defects.tolist()
                defects_list.sort(key=lambda x: x[0][3], reverse=True)
                s, e, f, d = defects_list[0][0]  # 得到的list中每项是[[x,y,z,k]]格式，所以需要0
                start = tuple(c[0][s][0])
                end = tuple(c[0][e][0])
                far = tuple(c[0][f][0])
                if (far[0] > 50 & far[1] > 30) | (far[0] < 50 & far[1] < 30):
                    cont[1] = 3
            else:
                return False, crop, [center_p, angle, cont[1], cont[2]]
            # print(far)
            # distant =
            # cv2.line(crop_wbg, start, end, [0, 255, 0], 2)
            # cv2.circle(crop_wbg, far, 5, [0, 0, 255], -1)
            # cv2.imshow(f'img{i[2]}', crop_wbg)
            # i[1] = self.classify_shape0(i)

        return True, crop, [center_p, angle, cont[1], cont[2]]
