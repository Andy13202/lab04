import cv2
import numpy as np

# 讀取圖片
image_path = "animal.jpg"  # 請更換為你的圖片路徑
image = cv2.imread(image_path)

# 轉換為灰階
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 去除雜訊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 邊緣檢測
edges = cv2.Canny(blurred, 50, 150)

# 擴展邊緣，使分割效果更好
kernel = np.ones((3, 3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=2)

# 建立遮罩
mask = cv2.threshold(edges_dilated, 0, 255, cv2.THRESH_BINARY)[1]

# 轉換遮罩為 3 通道
mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# 套用遮罩
result = cv2.bitwise_and(image, mask_3ch)

# 顯示結果
cv2.imshow("Original Image", image)
cv2.imshow("Removed Background", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
