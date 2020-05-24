import cv2
import numpy as np
import math


class detect(object):
    def __init__(self, image):
        self.image = image
        blurred = cv2.GaussianBlur(image, (3, 3), 0)  # 高斯模糊，降噪
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)  # 转为灰度图片
        autosobel = cv2.Canny(gray, 150, 200, 3)    # sobel变化，边缘检测
        contours, _ = cv2.findContours(autosobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detect_cont = self.filter_contours(contours)
        self.classified_cont = self.classify_cont(detect_cont)

    def filter_contours(self, con):    # 过滤得到疑似塑件的轮廓
        filtered = []
        square = [cv2.contourArea(x) for x in con]
        for i in range(len(square)):
            if 3500 < square[i] < 7500:
                filtered.append(con[i])
        return filtered

    def classify_cont(self, cont):    # 分类识别出的轮廓
        model_cont = np.load("./contdata.npy", allow_pickle=True)
        clsy_rlt = []     # 存放分类结果，格式[[轮廓，类型，区别度]...]
        for i, k in enumerate(model_cont):
            for j in cont:
                m = cv2.matchShapes(k, j, 1, 0.0)
                if m < 0.035:
                    clsy_rlt.append([j, k, m])
        clsy_rlt.sort(key=lambda x: x[2])
        return clsy_rlt    # 使用区分度排序

    def get_classified_pic(self):
        pic = self.image.copy()
        for i in self.classified_cont:
            cv2.drawContours(pic, [i[0]], -1, (0, 255, 0), 5)
        return True, pic    # 返回识别成功，识别后的图像


    def get_four_cont(self):
        return self.classified_cont[:4]    # 返回排序后的前四个轮廓

    def get_pic_info(self, cont):
        pics = []
        infos = []
        for i in cont:
            rect = cv2.minAreaRect(i[0])
            box = cv2.boxPoints(rect)  # 获得四个端点坐标
            box = np.int0(box)  # 变化为整型
            img = cv2.drawContours(self.image.copy(), [box], -1, (0, 255, 0), 5)
            # 找出中点，标出中心点
            center_p = (int(rect[0][0]), int(rect[0][1]))
            # 计算出长宽和倾斜角
            dst_w = int(math.floor(rect[1][0] / 2))
            dst_h = int(math.floor(rect[1][1] / 2))
            angle = rect[2]
            if dst_h > dst_w:
                dst_w, dst_h = dst_h, dst_w
                angle += 90
            M = cv2.getRotationMatrix2D(center_p, angle, 1.0)    # 获取旋转参数

            rotated = cv2.warpAffine(self.image, M, (img.shape[:2]))
            crop = rotated[center_p[1] - dst_h - 2:center_p[1] + dst_h + 2,
                   center_p[0] - dst_w - 2:center_p[0] + dst_w + 2]
            infos.append([center_p, angle, i[1], i[2]])
            pics.append(crop)
        return pics, infos



# def detect(image):
#     blurred = cv2.GaussianBlur(image, (3, 3), 0)  # 高斯模糊，降噪
#     blurred1 = cv2.pyrMeanShiftFiltering(image, 1, 60)
#     blurred2 = cv2.bilateralFilter(image, 9, 75, 75)  # 中值滤波
#     # blurred = image
#     cv2.imshow("blurred", blurred)
#     gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)  # 转为灰度图片
#
#     autosobel = cv2.Canny(gray, 150, 200, 3)
#
#     # 轮廓检测，返回轮廓list和轮廓关系list
#     contours, hierarchy = cv2.findContours(autosobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # realcont存储筛选结果
#     realcont = []
#     square = [cv2.contourArea(x) for x in contours]
#     for i in range(len(square)):
#         if 3500 < square[i] < 7500:
#             realcont.append(contours[i])
#             print(square[i])
#     print(f"找到的轮廓数目为 {len(realcont)}")
#
#     # 轮廓在白底中画出
#     white_bg = image.copy()
#     white_bg.fill(255)
#     wbg_cont = cv2.drawContours(white_bg, realcont, -1, (0, 255, 0), 3)
#     cv2.imshow("wbg_cont", wbg_cont)
#
#     # return True, wbg_cont
#
#     contours_cps = []  # 轮廓的中心点
#     contoursnamed = image.copy()
#     for i in range(len(realcont)):
#         rect = cv2.minAreaRect(realcont[i])
#         center_p = (int(rect[0][0]), int(rect[0][1]))
#         contours_cps.append(center_p)
#         contoursnamed = cv2.circle(contoursnamed, center_p, 4, (255, 0, 255), -1)
#         cv2.putText(contoursnamed, "contour{}".format(i), (center_p[0] - 20, center_p[1] - 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
#         cv2.imshow("contours named", contoursnamed)
#         if cv2.matchShapes(realcont[1], realcont[i], 1, 0.0) < 0.03:
#             print("contour 12 matched contour {}".format(i))
#
#     # 把检测到的轮廓分为3类
#     # 把三种被比较的样本轮廓保存，读取
#     # contdata = np.array([realcont[1], realcont[8], realcont[11]])
#     # np.save("./contdata.npy", contdata)
#     contdata = np.load("./contdata.npy", allow_pickle=True)
#     # print("sssss", contdata)
#     # print(type(contdata))
#     classified = image.copy()
#     for i, cont in enumerate(contdata):  # 8改为15用qt界面运行会数组越界？？？
#         for j in range(len(realcont)):
#             if cv2.matchShapes(cont, realcont[j], 1, 0.0) < 0.040:
#                 cv2.circle(classified, contours_cps[j], 4, (255, 0, 255), -1)
#                 cv2.putText(classified, "shape{}".format(i), (contours_cps[j][0] - 20, contours_cps[j][1] - 20),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
#     cv2.imshow("model match", classified)
#
#     clsy_rlt = []
#     for i, k in enumerate(contdata):  # 8改为15用qt界面运行会数组越界？？？
#         for j in realcont:
#             m = cv2.matchShapes(k, j, 1, 0.0)
#             if m < 0.035:
#                 clsy_rlt.append([j, i, m])
#     clsy_rlt.sort(key=lambda x: x[2])  # 使用区分度排序
#     print(len(clsy_rlt))
#     print("ssss", len(clsy_rlt))
#
#     # 缺陷检测 用来分别缺单角的图形
#     hull = cv2.convexHull(realcont[5], returnPoints=False)
#     defects = cv2.convexityDefects(realcont[5], hull)
#     print(defects)
#     img = image.copy()
#     for i in range(defects.shape[0]):
#         s, e, f, d = defects[9, 0]
#         start = tuple(realcont[5][s][0])
#         end = tuple(realcont[5][e][0])
#         far = tuple(realcont[5][f][0])
#         # distant =
#         cv2.line(img, start, end, [0, 255, 0], 2)
#         cv2.circle(img, far, 5, [0, 0, 255], -1)
#     cv2.imshow('img', img)
#
#
#
#
#
#
#     # 绘制轮廓的最小外接矩形，用来切割出需要的小图
#     rect = cv2.minAreaRect(realcont[5])
#     box = cv2.boxPoints(rect)  # 获得四个端点坐标
#     box = np.int0(box)  # 变化为整型
#     img3 = cv2.drawContours(image.copy(), [box], -1, (0, 255, 0), 5)
#     print(cv2.contourArea(realcont[6]))
#     cv2.imshow("test", img3)
#     # 找出中点，标出中心点
#     center_p = (int(rect[0][0]), int(rect[0][1]))
#
#     # 计算出长宽和倾斜角
#     print(rect[1][0], rect[1][1])
#     dst_w = int(math.floor(rect[1][0] / 2))
#     dst_h = int(math.floor(rect[1][1] / 2))
#     angle = rect[2]
#     if dst_h > dst_w:
#         dst_w, dst_h = dst_h, dst_w
#         angle += 90
#     print(dst_w, dst_h)
#
#     print(angle)
#
#     M = cv2.getRotationMatrix2D(center_p, angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (img3.shape[:2]))
#     crop = rotated[center_p[1] - dst_h - 2:center_p[1] + dst_h + 2,
#            center_p[0] - dst_w - 2:center_p[0] + dst_w + 2]
#
#     # 角点检测
#     crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     crop_gray = np.float32(crop_gray)
#     dst = cv2.cornerHarris(crop_gray, 4, 7, 0.04)
#     dst = cv2.dilate(dst, None)
#     # crop[dst > 0.01 * dst.max()] = [0, 0, 255]
#     cv2.imshow("coner", crop)
#     print(crop.shape)
#
#     corner = cv2.goodFeaturesToTrack(crop_gray, 6, 0.1, 10)
#     corner = np.int0(corner)
#     good_corner = crop.copy()
#     for i in corner:
#         x, y = i.ravel()
#         cv2.circle(good_corner, (x, y), 3, 255, -1)
#     cv2.imshow("goodcorner", good_corner)
#
#     # orb = cv2.ORB()
#     # kp1, des1 = orb.detectAndCompute(corner, 2)
#     # kp2, des2 = orb.detectAndCompute(wbg_cont, None)
#     # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     # match = bf.match(des1, des2)
#     # matches = sorted(match, key = lambda x : x.distance)
#     # img4 = cv2.drawMatchesKnn(corner, kp1, wbg_cont, kp2, matches[:10], flags=2)
#     # cv2.imshow("match test", img4)
#
#     # 标出中心点并标名
#     dis_c_dot = cv2.circle(rotated, center_p, 4, (255, 0, 255), -1)
#     cv2.putText(dis_c_dot, "center", (center_p[0] - 20, center_p[1] - 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
#
#     cv2.imshow("ROI", crop)
#     cv2.imshow("rotated", rotated)
#
#     cv2.imshow("centre point", dis_c_dot)
#
#     img2 = cv2.drawContours(image.copy(), contours, -1, (0, 0, 255), 1)
#     cv2.imshow("img2", img2)
#     return True, img2
#

def findcontour(image):
    # 均值漂移,中和色彩分布相近的颜色，平滑色彩细节，侵蚀掉面积较小的颜色区域
    blurred1 = cv2.pyrMeanShiftFiltering(image, 1, 60)
    gray1 = cv2.cvtColor(blurred1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otus二值化，自计算优阈值
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(thr_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    square = [cv2.contourArea(x) for x in contours]
    realcont = []
    for i in range(len(square)):
        if square[i] > 3500:
            realcont.append(contours[i])
    # realcont = [if x <3000: 0 else: x for x in contours]
    print(realcont)
    img = cv2.drawContours(image, realcont, -1, (0, 255, 0), 1)
    cv2.imshow("img", img)
    print(square)

    pass


if __name__ == "__main__":
    src = cv2.imread("C:/Users/62329/Desktop/object.jpg")
    src1 = cv2.imread("C:/Users/62329/Desktop/test2.jpg")
    cv2.imshow("origin", src)
    detect(src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
