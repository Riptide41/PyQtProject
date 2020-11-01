import cv2
import numpy as np
import math, time
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from math import sqrt


# **********************************************************************
# 函数名称：normalization
# 函数参数：data_arr：需要归一化的array
# 返回参数：归一化后的array
# 函数功能：将传入的array做归一化后返回
# **********************************************************************
def normalization(data_arr):
    _range = np.max(data_arr) - np.min(data_arr)
    return (data_arr - np.min(data_arr)) / _range


class Detect(object):
    def __init__(self):
        # 载入分类使用的模型
        self.model = load_model("./model_1020_2.h5")

    # **********************************************************************
    # 函数名称：image_process
    # 函数参数：image：需要处理的图片
    #         min_square:检测的最小面积
    #         max_square:检测的最大面积
    # 返回参数：无
    # 函数功能：将传入的image做处理，识别出轮廓
    # **********************************************************************
    def image_process(self, image, min_square=50, max_square=5000):
        self.image = image
        self.max_distance = sqrt((self.image.shape[0] / 2) ** 2 + self.image.shape[1] ** 2)
        # cv2.imshow("ooooooooooo", self.image)
        cv2.imwrite(f"./conts/ori.jpg", self.image)
        # blurred = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯模糊，降噪w

        # blurred = cv2.bilateralFilter(blurred, 0, 100, 15)
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图片

        med_blurred = cv2.medianBlur(self.gray, 5, 0)  # 中值模糊，降噪，保留边缘信息
        cv2.imwrite(f"./conts/denoise.jpg", med_blurred)
        # 锐化图片
        sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpen_image = cv2.filter2D(med_blurred, cv2.CV_32F, sharpen_op)
        sharpen_image = cv2.convertScaleAbs(sharpen_image)
        # cv2.imshow("sharpen", sharpen_image)

        cv2.imwrite(f"./conts/sharpened.jpg", sharpen_image)  # tiaoshi
        sharpen_canny = cv2.Canny(sharpen_image, 150, 200)
        # cv2.imshow("sharpen_canny", sharpen_canny)
        cv2.imwrite(f"./conts/canny.jpg", sharpen_canny)

        # binary = cv2.bitwise_not(sharpen_image)
        # ret, binary = cv2.threshold(binary, 50, 0,  cv2.THRESH_TOZERO)
        # cv2.imshow("ttttest", binary)

        ret, binary = cv2.threshold(sharpen_image, 40, 255, cv2.THRESH_BINARY)  # 图片二值化
        # 图片开操作，清除小黑点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 边缘检测
        # cv2.imshow("binary pic", binary)
        autosobel = cv2.Canny(binary, 100, 250, 3)  # sobel变化，边缘检测
        # cv2.imshow(" pic", autosobel)

        self.contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # opencv3返回三参数

        self.detect_min_square = min_square
        self.detect_max_square = max_square
        self.detect_cont = self.filter_contours()
        self.classified_cont = self.classify_cont()

    # **********************************************************************
    # 函数名称：filter_contours
    # 函数参数：无
    # 返回参数：返回使用面积过滤后的轮廓
    # 函数功能：使用面积对轮廓进行排除
    # **********************************************************************
    def filter_contours(self):  # 过滤得到疑似塑件的轮廓
        filtered = []
        square = [cv2.contourArea(x) for x in self.contours]
        for i in range(len(square)):
            # if self.detect_min_square < self.contours[i].size < self.detect_max_square:
            if self.detect_min_square < square[i] < self.detect_max_square:
                # print(self.contours[i].size)
                # print("area:", cv2.contourArea(self.contours[i]))
                filtered.append(self.contours[i])

        # ******************测试使用***********************
        # test = cv2.drawContours(self.image.copy(), [filtered[16]], -1, 255, 5)
        # print("test size:", filtered[3].size)
        # model_cont = np.load("./class_0.npy", allow_pickle=True)
        # model_cont[2] = filtered[13]
        # np.save("./class_0.npy", model_cont)

        # cv2.imshow("test", test)
        return filtered

    # **********************************************************************
    # 函数名称：classify_cont
    # 函数参数：无
    # 返回参数：返回list，[[轮廓, 类型, 区分度]....]
    # 函数功能：对轮廓与模板轮廓进行匹配，及初步识别返回区分度
    # **********************************************************************
    def classify_cont(self):  # 分类识别出的轮廓
        model_cont = np.load("class_0.npy", allow_pickle=True)  # 读取已保存的轮廓信息
        clsy_rlt = []  # 存放分类结果，格式[[轮廓，类型，区别度]...]
        cnt_image = self.image.copy()
        for j in self.detect_cont:
            d_max = 1
            rlt = 0
            for i, k in enumerate(model_cont):
                # 对轮廓与模板轮廓进行比对
                m = cv2.matchShapes(k, j, 1, 0.0)
                if m < d_max:
                    d_max = m
                    rlt = [j, i, d_max]  # [轮廓, 类型, 区分度]
            cv2.drawContours(cnt_image, [j], -1, (0, 133, 233), 2)
            if rlt:
                clsy_rlt.append(rlt)
        cv2.imwrite("./conts/cont.jpg", cnt_image)
        return clsy_rlt  # 使用区分度排序

    # 在原图上绘制轮廓
    # def get_classified_pic(self):
    #     pic = self.image.copy()
    #     for i, c in enumerate(self.classified_cont):
    #         if i < 4:  # 前三个标绿，后面标红
    #             cv2.drawContours(pic, [c[0]], -1, (0, 255, 0), 5)
    #         else:
    #             cv2.drawContours(pic, [c[0]], -1, (0, 0, 255), 5)
    #     return True, pic  # 返回识别成功，识别后的图像

    # **********************************************************************
    # 函数名称：classify_cont
    # 函数参数：无
    # 返回参数：返回处理是否成功、[[中心点坐标,旋转角度,塑件类型,分数,单塑件图片,此塑件轮廓]...]、对前四轮廓标出的全图
    # 函数功能：对每个轮廓进行处理，提取其中的信息并排序后返回。
    # **********************************************************************
    def get_four_pic_info(self):
        infos_and_pics = []
        result = []
        # self.all_pics = []
        full_pic = self.image.copy()
        for i in self.classified_cont:
            flag, info_and_pic = self.get_pic_info(i)
            if flag:
                infos_and_pics.append(info_and_pic)
            font = cv2.FONT_HERSHEY_SIMPLEX
            full_pic = cv2.putText(full_pic, f"{'%.1f' % info_and_pic[3]}",
                                   (info_and_pic[0][0] - 30, info_and_pic[0][1]), font, 0.8, (0, 121, 233), 2)
        count = 1
        infos_and_pics.sort(key=lambda x: x[3], reverse=True)  # 使用分数对塑件进行排序
        for i in infos_and_pics:
            # # **************************测试*********************
            # fla, pi, inf = self.get_pic_info(i)
            # if fla:
            #     self.all_pics.append(pi)
            # ***************************************************
            if count > 4:
                cv2.drawContours(full_pic, [i[5]], -1, (0, 0, 255), 2)
                # count += 1
                # continue
            else:
                cv2.drawContours(full_pic, [i[5]], -1, (0, 255, 0), 2)  # 入选标绿
                result.append(i)  # [中心点坐标，倾斜角度，类型，区别度]
                count += 1
        while count < 4:
            result.append([("NaN", "NaN"), "NaN", "NaN", "NaN", None])
            count += 1
        return True, result, full_pic  # 返回排序后的前四个个轮廓

    # **********************************************************************
    # 函数名称：get_pic_info
    # 函数参数：要进行分析的轮廓
    # 返回参数：返回处理是否成功、[中心点坐标,旋转角度,塑件类型,分数,单塑件图片,此塑件轮廓]
    # 函数功能：对每个轮廓进行处理，提取其中的信息并排序后返回。
    # **********************************************************************
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
        rotated = cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]))
        crop = rotated[center_p[1] - dst_h - 2:center_p[1] + dst_h + 2,
               center_p[0] - dst_w - 2:center_p[0] + dst_w + 2]

        # 当塑件处于边缘时，截取的图片size为0，抛弃
        if crop.size is 0:
            return False, [center_p, angle, cont[1], cont[2], crop]
        # if not self.detect_min_square < len(crop) * len(crop[0]) < self.detect_max_square:
        #     return False, crop, [center_p, angle, cont[1], cont[2]]
        pred_pic = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        pred_pic = cv2.resize(pred_pic, (204, 114))
        pred_input = pred_pic[np.newaxis, :, :, np.newaxis]
        # cv2.imshow("kkkklsd",crop)
        pred_y = self.model.predict(normalization(pred_input))
        # print(pred_y)
        # print(np.argmax([pred_y]), cont[1])
        cont_type = np.argmax([pred_y])  # 定义轮廓类型为CNN结果
        score = self.score_pieces(center_p, cont)
        return True, [center_p, angle, cont_type, score, crop, cont[0]]

    # **********************************************************************
    # 函数名称：score_pieces
    # 函数参数：中心点坐标，轮廓
    # 返回参数：返回分数
    # 函数功能：对轮廓进行打分
    # **********************************************************************
    def score_pieces(self, centre, cont):
        # print(self.image.shape)
        distance_sco = sqrt(centre[1] ** 2 + (centre[0]) ** 2) * 100 / self.max_distance
        # print(distance_sco)
        # print(self.max_distance)
        distinguish_sco = cont[2] * 100
        # print("diswsco", distinguish_sco)
        score = 100 - distance_sco * 0.3 - 0.7 * distinguish_sco
        # print(score)
        return score

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

    # def four_pic(self):
    #     for cont in self.classified_cont:
    #         print("conts", cont)
    #         rect = cv2.minAreaRect(cont[0])
    #         cv2.boxPoints(rect)  # 获得四个端点坐标
    #         # img = cv2.drawContours(self.image.copy(), [box], -1, (0, 255, 0), 5)
    #         # 找出中点，标出中心点
    #         center_p = (int(rect[0][0]), int(rect[0][1]))
    #         # 计算出长宽和倾斜角
    #         dst_w = int(math.floor(rect[1][0] / 2))
    #         dst_h = int(math.floor(rect[1][1] / 2))
    #         angle = rect[2]
    #         if dst_h > dst_w:
    #             dst_w, dst_h = dst_h, dst_w
    #             angle += 90
    #         print("dest h w:", dst_h, dst_w)
    #         M = cv2.getRotationMatrix2D(center_p, angle, 1.0)  # 获取旋转参数
    #         # print("sssss", self.image.shape[0:2])
    #         rotated = cv2.warpAffine(self.gray, M, (1623, 1080))
    #         crop = rotated[center_p[1] - dst_h - 2:center_p[1] + dst_h + 2,
    #                center_p[0] - dst_w - 2:center_p[0] + dst_w + 2]
    #         if crop.size is 0:
    #             return False, crop, [center_p, angle, cont[1], cont[2]]
    #         cv2.imshow("kkkkl",crop)
    #         print(crop.shape)
    #         crop = cv2.resize(crop, (204, 114))
    #         crop = crop[np.newaxis, :, :, np.newaxis]
    #         # cv2.imshow("kkkklsd",crop)
    #         pred_y = self.model.predict(normalization(crop))
    #         print(pred_y)
