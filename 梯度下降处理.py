import cv2 as cv
import numpy as np

def custom_threshold(image,index):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w*h])
    print("m.sum = %s" %(m.sum()))
    mean = m.sum() / (w*h)
    print("mean : ", mean)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.imshow("custom_threshold_%s" %index, binary)

def lapalian_demo(image):
    #dst = cv.Laplacian(image, cv.CV_32F)
    #lpls = cv.convertScaleAbs(dst)
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    dst = cv.filter2D(image, cv.CV_32F, kernel=kernel)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("lapalian_demo", lpls)


def sobel_demo(image):
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient-x", gradx)
    cv.imshow("gradient-y", grady)

    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("gradient", gradxy)
    return gradxy



print("--------- Python OpenCV Tutorial ---------")
src = cv.imread("./images/2.BMP")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
# lapalian_demo(src)
img = sobel_demo(src)
custom_threshold(img, 1)
cv.waitKey(0)

cv.destroyAllWindows()
