import cv2
import numpy as np
import math

def canny(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)  # 高斯模糊，降噪
    blurred1 = cv2.pyrMeanShiftFiltering(image, 1, 60)
    blurred2 = cv2.bilateralFilter(image, 9, 75, 75)  # 中值滤波
    # blurred = image
    cv2.imshow("blurred", blurred)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)  # 转为灰度图片
    gray1 = cv2.cvtColor(blurred1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(blurred2, cv2.COLOR_BGR2GRAY)

    # x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)  # CV_16S表示16位有符号整形
    # y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    # absX = cv2.convertScaleAbs(x)  # 把越界的数值转化回255uint8
    # bsY = cv2.convertScaleAbs(y)
    autosobel = cv2.Canny(gray, 150, 200, 3)
    manualsobel = cv2.Canny(gray1, 150, 200)
    autosobel2 = cv2.Canny(gray2, 150, 200)
    _, thresh = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("bilaFIl", autosobel2)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    # print(autosobel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morgra = cv2.morphologyEx(gray1, cv2.MORPH_GRADIENT, kernel)
    thresh2 = cv2.adaptiveThreshold(morgra, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    cv2.imshow("MORGRA", thresh2)

    # cv2.imshow("drawContours", thresh)
    cv2.imshow("AutoSobel", autosobel)
    cv2.imshow("ManualSobel", manualsobel)
    # cv2.imshow("Edge2 output", img_opening)

    # 直线检测，不好用
    '''
    minLineLength = 1
    maxLineGap = 1
    lines = cv2.HoughLinesP(autosobel, 1, np.pi / 180, 2, minLineLength, maxLineGap)
    print(lines.shape)

    linepic = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(linepic, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.imshow("linepic", linepic)
    '''
    # 轮廓检测，返回轮廓list和轮廓关系list
    contours, hierarchy = cv2.findContours(autosobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # realcont存储筛选结果
    realcont = []
    square = [cv2.contourArea(x) for x in contours]
    for i in range(len(square)):
        if 3500 < square[i] < 7500:
            realcont.append(contours[i])
            print(square[i])
    print(f"找到的轮廓数目为 {len(realcont)}")

    # 轮廓在白底中画出
    white_bg = image.copy()
    white_bg.fill(255)
    wbg_cont = cv2.drawContours(white_bg, realcont, -1, (0, 255, 0), 3)
    cv2.imshow("wbg_cont", wbg_cont)

    contours_cps = []    # 轮廓的中心点
    contoursnamed = image.copy()
    for i in range(len(realcont)):
        rect = cv2.minAreaRect(realcont[i])
        center_p = (int(rect[0][0]), int(rect[0][1]))
        contours_cps.append(center_p)
        contoursnamed = cv2.circle(contoursnamed, center_p, 4, (255, 0, 255), -1)
        cv2.putText(contoursnamed, "contour{}".format(i), (center_p[0] - 20, center_p[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.imshow("contours named", contoursnamed)
        if cv2.matchShapes(realcont[1], realcont[i], 1, 0.0) < 0.03:
            print("contour 12 matched contour {}".format(i))

    # 把检测到的轮廓分为3类
    classified = image.copy()
    for i in (1, 11, 15):
         for j in range(len(realcont)):
             if cv2.matchShapes(realcont[i], realcont[j], 1, 0.0) < 0.035:
                 cv2.circle(classified, contours_cps[j], 4, (255, 0, 255), -1)
                 cv2.putText(classified, "shape{}".format(i), (contours_cps[j][0] - 20, contours_cps[j][1] - 20),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.imshow("model match", classified)

    # 绘制轮廓的最小外接矩形，用来切割出需要的小图
    rect = cv2.minAreaRect(realcont[1])
    box = cv2.boxPoints(rect)    # 获得四个端点坐标
    box = np.int0(box)    # 变化为整型
    img3 = cv2.drawContours(image.copy(), [box], -1, (0, 255, 0), 5)
    print(cv2.contourArea(realcont[6]))
    cv2.imshow("test", img3)
    # 旋转矩形
    # 找出中点，标出中心点
    center_p = (int(rect[0][0]), int(rect[0][1]))


    # 计算出长宽和倾斜角
    print(rect[1][0], rect[1][1])
    dst_w = int(math.floor(rect[1][0]/2))
    dst_h = int(math.floor(rect[1][1]/2))
    angle = rect[2]
    if dst_h > dst_w:
        dst_w, dst_h = dst_h, dst_w
        angle += 90
    print(dst_w, dst_h)

    print(angle)

    M = cv2.getRotationMatrix2D(center_p, angle, 1.0)
    h, w = img3.shape[:2]
    rotated = cv2.warpAffine(wbg_cont, M, (w, h))
    crop = rotated[center_p[1]-dst_h-2:center_p[1]+dst_h+2,
           center_p[0]-dst_w-2:center_p[0]+dst_w+2]

    # 角点检测
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_gray = np.float32(crop_gray)
    dst = cv2.cornerHarris(crop_gray, 4, 7, 0.04)
    dst = cv2.dilate(dst, None)
    #crop[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow("coner", crop)
    print(crop.shape)

    corner = cv2.goodFeaturesToTrack(crop_gray, 6, 0.1, 10)
    corner = np.int0(corner)
    good_corner = crop.copy()
    for i in corner:
        x, y = i.ravel()
        cv2.circle(good_corner,(x, y), 3, 255, -1)
    cv2.imshow("goodcorner", good_corner)

    # orb = cv2.ORB()
    # kp1, des1 = orb.detectAndCompute(corner, 2)
    # kp2, des2 = orb.detectAndCompute(wbg_cont, None)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # match = bf.match(des1, des2)
    # matches = sorted(match, key = lambda x : x.distance)
    # img4 = cv2.drawMatchesKnn(corner, kp1, wbg_cont, kp2, matches[:10], flags=2)
    # cv2.imshow("match test", img4)

    # 标出中心点并标名
    dis_c_dot = cv2.circle(rotated, center_p, 4, (255, 0, 255), -1)
    cv2.putText(dis_c_dot, "center", (center_p[0] - 20, center_p[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    cv2.imshow("ROI", crop)
    cv2.imshow("rotated", rotated)

    cv2.imshow("centre point", dis_c_dot)

    img2 = cv2.drawContours(image.copy(), contours, -1, (0, 0, 255), 1)
    cv2.imshow("img2", img2)


def watershed(image):
    # 均值漂移,中和色彩分布相近的颜色，平滑色彩细节，侵蚀掉面积较小的颜色区域
    blurred1 = cv2.pyrMeanShiftFiltering(image, 1, 60)
    gray1 = cv2.cvtColor(blurred1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otus二值化，自计算优阈值
    cv2.imshow("thresh", thresh)
    # 开操作，去除白噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow("afteropen", thr_open)
    # 确认后景区域
    sure_bg = cv2.morphologyEx(thr_open, cv2.MORPH_DILATE, kernel, iterations=1)
    # 距离变换，确认前景区域
    dist_transform = cv2.distanceTransform(thr_open, cv2.DIST_L2, 0)
    _, dist_output = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    cv2.imshow("sure_bg", sure_bg)
    cv2.imshow("dist_output", dist_output)
    sure_fg = np.uint8(dist_output)
    # 确认未知区域
    unknow = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)  # 标记前后景区域
    # 上述函数吧背景标记成0，分水岭函数需要背景是1,未知改为0
    markers += 1
    markers[unknow == 255] = 0
    # 分水岭算法
    markers = cv2.watershed(image, markers)
    dst1 = np.zeros(markers.shape)
    dst1[markers == -1] = 255
    cv2.imshow("dst0", dst1)  # 显示结果
    image[markers == -1] = [0, 0, 255]
    cv2.imshow("dst", image)  # 在原图显示结果


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


def test(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    threshhold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    cv2.imshow("two", threshhold)
    thr_blurred = cv2.GaussianBlur(threshhold, (5, 5), 0)

    _, threshhold1 = cv2.threshold(thr_blurred, 200, 255, cv2.THRESH_BINARY)
    # img_opening_blurred = threshhold1
    # '''
    cv2.imshow("threshhold1", threshhold1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 取3*3正方体为核
    img_close = cv2.morphologyEx(threshhold1, cv2.MORPH_CLOSE, kernel)  # 调用形态学操作API
    contours, hierarchy = cv2.findContours(img_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 1)

    cv2.imshow("drawContours", image)
    cv2.imshow("img_close", img_close)

if __name__ == "__main__":
    src = cv2.imread("C:/Users/62329/Desktop/object.jpg")
    src1 = cv2.imread("C:/Users/62329/Desktop/test2.jpg")
    cv2.imshow("origin", src)
    canny(src)
    #watershed(src1)
    # cv2.imshow("absY", absY)
    # cv2.imshow("output", absX)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
