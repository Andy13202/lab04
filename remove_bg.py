import cv2
import numpy as np

# 讀取圖片
image_path = "person.jpg"  # 請更換為你的圖片
image = cv2.imread(image_path)

# 取得圖片尺寸
height, width = image.shape[:2]

# 建立遮罩
mask = np.zeros((height, width), np.uint8)

# 建立背景與前景模型（GrabCut 需要的格式）
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# 設定 ROI（感興趣區域），假設人物位於圖像中央
rect = (10, 10, width - 20, height - 20)

# 應用 GrabCut（基於矩形區域進行初步前景分割）
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# 生成前景遮罩（0 和 2 為背景，1 和 3 為前景）
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

# 將人物與背景分離
result = image * mask2[:, :, np.newaxis]

# 轉換為 4 通道（RGBA），將背景變為透明
b, g, r = cv2.split(result)
alpha = mask2 * 255  # 透明度通道
result_rgba = cv2.merge([b, g, r, alpha])

# 儲存為 PNG（支援透明背景）
output_path = "person_no_bg.png"
cv2.imwrite(output_path, result_rgba)

# 顯示結果
cv2.imshow("Original", image)
cv2.imshow("Removed Background", result_rgba)

cv2.waitKey(0)
cv2.destroyAllWindows()
