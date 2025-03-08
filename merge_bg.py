import cv2
import numpy as np

# 讀取去背後的圖片（有透明背景）
foreground_path = "person_no_bg.png"  # 請更換為你的去背後圖片
foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)  # 讀取 RGBA 格式

# 讀取新背景圖片
background_path = "new_background.jpg"  # 請更換為你的背景圖片
background = cv2.imread(background_path)

# 確保背景和前景大小相符（調整背景尺寸）
fg_h, fg_w = foreground.shape[:2]
background = cv2.resize(background, (fg_w, fg_h))

# 分離前景的 RGB 與 Alpha 通道
b, g, r, alpha = cv2.split(foreground)

# 轉換 Alpha 通道為 0~1 的遮罩
alpha = alpha / 255.0

# 前景與背景合成
foreground_rgb = cv2.merge([b, g, r])
blended = (foreground_rgb * alpha[:, :, np.newaxis]) + (background * (1 - alpha[:, :, np.newaxis]))
blended = blended.astype(np.uint8)  # 確保為整數格式

# 儲存並顯示結果
output_path = "person_with_new_bg.png"
cv2.imwrite(output_path, blended)

cv2.imshow("Replaced Background", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
