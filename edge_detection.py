import cv2
import numpy as np

# 讀取灰階圖像
img = cv2.imread("請更換為你的圖片路徑", cv2.IMREAD_GRAYSCALE)

# 計算 Sobel 梯度
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # X 方向梯度
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Y 方向梯度
sobel = cv2.magnitude(sobel_x, sobel_y)  # 計算梯度幅值

# 設定閥值 (例如 100)
threshold_value = 50
_, sobel_thresholded = cv2.threshold(sobel, threshold_value, 255, cv2.THRESH_BINARY)

# 轉換為 8-bit 顯示格式
sobel_thresholded = np.uint8(sobel_thresholded)

# 顯示結果
cv2.imshow("Sobel Thresholded", sobel_thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()
