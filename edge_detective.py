import cv2
import numpy as np
from ipywidgets import interact

def sobel_edge_detection(threshold_value):
    # 讀取灰階圖像
    img = cv2.imread("original.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 計算 Sobel 梯度
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)  

    # 設定閥值
    _, sobel_thresholded = cv2.threshold(sobel, threshold_value, 255, cv2.THRESH_BINARY)
    
    # 轉換為 8-bit 格式
    sobel_thresholded = np.uint8(sobel_thresholded)

    # 顯示結果
    cv2.imshow("Sobel Edge Detection", sobel_thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 讓學生調整 threshold
interact(sobel_edge_detection, threshold_value=(0, 255, 5))
