import cv2
import numpy as np
import math, time
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def normalization(data_arr):
    _range = np.max(data_arr) - np.min(data_arr)
    return (data_arr - np.min(data_arr)) / _range

class Detect(object):
    def __init__(self):
        self.model = load_model("./1008model_2_3.h5")

    def image_process(self, image, min_square=50, max_square=5000):
        self.image = image
        cv2.imshow("ooooooooooo", self.image)
        # blurred = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯模糊，降噪w

        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
        # blurred = cv2.filter2D(image, -1, kernel=kernel)
        # cv2.imshow("ruihua", blurred)
        # blurred = cv2.bilateralFilter(blurred, 0, 100, 15)
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图片

        med_blurred = cv2.medianBlur(self.gray, 5, 0)  # 中值模糊，降噪，保留边缘信息
        # surf = cv2.xfeatures2d.SURF_create(30000)
        # kp = surf.detect(med_blurred, None)
        # img = cv2.drawKeypoints(med_blurred, kp, None, (255, 0, 0), 4)
        # cv2.imshow("sss", img)
        # ssr_re = singleScaleRetinex(med_blurred, 0)
        # cv2.imshow("ori", self.gray)

        # 锐化图片
        sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpen_image = cv2.filter2D(med_blurred, cv2.CV_32F, sharpen_op)
        sharpen_image = cv2.convertScaleAbs(sharpen_image)
        cv2.imshow("sharpen", sharpen_image)
        # sharpen_canny = cv2.Canny(sharpen_image, 150, 200)
        # cv2.imshow("sharpen_canny", sharpen_canny)

        ret, binary = cv2.threshold(sharpen_image, 120, 255, cv2.THRESH_BINARY)
        # binary = cv2.bitwise_not(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 边缘检测
        cv2.imshow("binary pic", binary)
        # autosobel = cv2.Canny(binary, 150, 200, 3)  # sobel变化，边缘检测
        # cv2.imshow(" pic", autosobel)

        # _, self.contours, _ = cv2.findContours(autosobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # opencv3返回三参数
        self.contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # opencv4返回三参数


        self.detect_min_square = min_square
        self.detect_max_square = max_square
        self.detect_cont = self.filter_contours()
        self.classified_cont = self.classify_cont()

        # self.four_pic()

    def filter_contours(self):  # 过滤得到疑似塑件的轮廓
        filtered = []
        square = [cv2.contourArea(x) for x in self.contours]
        for i in range(len(square)):
            # if self.detect_min_square < self.contours[i].size < self.detect_max_square:
            if self.detect_min_square < square[i] < self.detect_max_square:
                print(self.contours[i].size)
                print("area:", cv2.contourArea(self.contours[i]))
                filtered.append(self.contours[i])

        # ******************测试使用***********************
        # test = cv2.drawContours(self.image.copy(), [filtered[16]], -1, 255, 5)
        # print("test size:", filtered[3].size)
        # model_cont = np.load("./class_0.npy", allow_pickle=True)
        # model_cont[2] = filtered[13]
        # np.save("./class_0.npy", model_cont)

        # cv2.imshow("test", test)
        return filtered

    def classify_cont(self):  # 分类识别出的轮廓
        model_cont = np.load("class_0.npy", allow_pickle=True)  # 读取已保存的轮廓信息
        clsy_rlt = []  # 存放分类结果，格式[[轮廓，类型，区别度]...]
        cnt_image = self.image.copy()
        for j in self.detect_cont:
            for i, k in enumerate(model_cont):

                m = cv2.matchShapes(k, j, 1, 0.0)
                # print("detected aread:", j.size)
                # cv2.drawContours(self.image, [j], -1, (0, 255, 0), 5)


                if m < 0.065:
                    cnt_image = cv2.drawContours(cnt_image, [j], -1, (0, 255, 0), 5)
                    clsy_rlt.append([j, i, m])
                    break

        cv2.imshow("sdsdddddd", cnt_image)

        clsy_rlt.sort(key=lambda x: x[2])
        return clsy_rlt  # 使用区分度排序

    # 在原图上绘制轮廓
    def get_classified_pic(self):
        pic = self.image.copy()
        for i, c in enumerate(self.classified_cont):
            if i < 4:  # 前三个标绿，后面标红
                cv2.drawContours(pic, [c[0]], -1, (0, 255, 0), 5)
            else:
                cv2.drawContours(pic, [c[0]], -1, (0, 0, 255), 5)
        return True, pic  # 返回识别成功，识别后的图像

    # 获取四个轮廓信息，并在原图上绘制轮廓
    def get_four_pic_info(self):
        pics = []
        infos = []
        self.all_pics = []
        full_pic = self.image.copy()
        for i in self.classified_cont:
            # # **************************测试*********************
            # fla, pi, inf = self.get_pic_info(i)
            # if fla:
            #     self.all_pics.append(pi)
            # ***************************************************
            if len(pics) >= 4:
                cv2.drawContours(full_pic, [i[0]], -1, (0, 0, 255), 5)
                continue
            flag, pic, info = self.get_pic_info(i)
            if flag:
                cv2.drawContours(full_pic, [i[0]], -1, (0, 255, 0), 6)  # 入选标绿
                infos.append(info)  # [中心点坐标，倾斜角度，类型，区别度]
                pics.append(pic)
            else:
                cv2.drawContours(full_pic, [i[0]], -1, (0, 0, 255), 5)
        while len(pics) < 4:
            pics.append(cv2.imread("./Faildetect.png"))
            infos.append([("NaN", "NaN"), "NaN", "NaN", "NaN"])
        # print("cont_num:", cont_num)
        return True, pics, infos, full_pic  # 返回排序后的前四个个轮廓

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
        print("!!dest h w:", dst_h, dst_w)
        M = cv2.getRotationMatrix2D(center_p, angle, 1.0)  # 获取旋转参数
        # print("sssss", self.image.shape[0:2])
        rotated = cv2.warpAffine(self.image, M, (1623, 1080))
        crop = rotated[center_p[1] - dst_h - 2:center_p[1] + dst_h + 2,
               center_p[0] - dst_w - 2:center_p[0] + dst_w + 2]
        # 当塑件处于边缘时，截取的图片size为0，抛弃
        print()
        if crop.size is 0:
            return False, crop, [center_p, angle, cont[1], cont[2]]
        # if not self.detect_min_square < len(crop) * len(crop[0]) < self.detect_max_square:
        #     return False, crop, [center_p, angle, cont[1], cont[2]]
        pred_pic = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        pred_pic = cv2.resize(pred_pic, (204, 114))
        pred_input = pred_pic[np.newaxis, :, :, np.newaxis]
        # cv2.imshow("kkkklsd",crop)
        pred_y = self.model.predict(normalization(pred_input))
        print(np.argmax([pred_y]), cont[1])
        cont[1] = np.argmax(pred_y)
        # if crop.empty():
        #     print("error occur")

        # # 区分shape2
        # if cont[1] == 2:
        #     # 在白底上绘制轮廓
        #     w_bg = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        #     w_bg.fill(0)
        #     cv2.drawContours(w_bg, [cont[0]], -1, 255, 2)
        #     # 旋转使轮廓水平
        #     rotated_wbg = cv2.warpAffine(w_bg, M, (self.image.shape[0:2]))
        #     crop_wbg = rotated_wbg[center_p[1] - dst_h - 5:center_p[1] + dst_h + 5,
        #                center_p[0] - dst_w - 5:center_p[0] + dst_w + 5]
        #     # 检测轮廓，得到轮廓list（只有一个）
        #     # cv2.imshow("ss", crop_wbg)
        #     # _, c, _ = cv2.findContours(crop_wbg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     c, _ = cv2.findContours(crop_wbg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     if c:
        #         # sss = cv2.cvtColor(crop_wbg.copy(), cv2.COLOR_GRAY2BGR)
        #         # sss = cv2.drawContours(sss, [c[0]], -1, (0,255,0), 1)
        #         # cv2.imshow("www", sss)
        #         # 缺陷检测，检测缺口的位置，即距离凸包最远的点
        #         hull = cv2.convexHull(c[0], returnPoints=False)
        #         # print(c[0])
        #         defects = cv2.convexityDefects(c[0], hull)
        #         # print(defects)
        #         defects_list = defects.tolist()
        #         defects_list.sort(key=lambda x: x[0][3], reverse=True)
        #         s, e, f, d = defects_list[0][0]  # 得到的list中每项是[[x,y,z,k]]格式，所以需要0
        #         start = tuple(c[0][s][0])
        #         end = tuple(c[0][e][0])
        #         far = tuple(c[0][f][0])
        #         if (far[0] > 50 & far[1] > 30) | (far[0] < 50 & far[1] < 30):
        #             cont[1] = 3
        #     else:
        #         # *********************ceshi*************************Falsec才对
        #         return True, crop, [center_p, angle, cont[1], cont[2]]
        #     # print(far)
        #     # distant =
        #     # cv2.line(crop_wbg, start, end, [0, 255, 0], 2)
        #     # cv2.circle(crop_wbg, far, 5, [0, 0, 255], -1)
        #     # cv2.imshow(f'img{i[2]}', crop_wbg)
        #     # i[1] = self.classify_shape0(i)

        return True, crop, [center_p, angle, cont[1], cont[2]]

    def four_pic(self):
        for cont in self.classified_cont:
            print("conts", cont)
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
            print("dest h w:", dst_h, dst_w)
            M = cv2.getRotationMatrix2D(center_p, angle, 1.0)  # 获取旋转参数
            # print("sssss", self.image.shape[0:2])
            rotated = cv2.warpAffine(self.gray, M, (1623, 1080))
            crop = rotated[center_p[1] - dst_h - 2:center_p[1] + dst_h + 2,
                   center_p[0] - dst_w - 2:center_p[0] + dst_w + 2]
            if crop.size is 0:
                return False, crop, [center_p, angle, cont[1], cont[2]]
            cv2.imshow("kkkkl",crop)
            print(crop.shape)
            crop = cv2.resize(crop, (204, 114))
            crop = crop[np.newaxis, :, :, np.newaxis]
            # cv2.imshow("kkkklsd",crop)
            pred_y = self.model.predict(normalization(crop))
            print(pred_y)

