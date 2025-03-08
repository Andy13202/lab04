import cv2
import numpy as np

# 讀取圖片
image_path = "animal.jpg"  # 請更換為你的圖片路徑
image = cv2.imread(image_path)

# 創建遮罩
mask = np.zeros(image.shape[:2], np.uint8)

# 建立背景與前景模型（GrabCut 需要的格式）
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# 設定 ROI（感興趣區域），假設動物大致位於圖像中央
height, width = image.shape[:2]
rect = (10, 10, width - 20, height - 20)  # 預設 ROI 範圍

# 應用 GrabCut
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# 產生前景遮罩
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

# 應用遮罩
result = image * mask2[:, :, np.newaxis]

# 顯示結果
cv2.imshow("Original Image", image)
cv2.imshow("Removed Background", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
