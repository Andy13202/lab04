from rembg import remove
import cv2
import numpy as np

# 讀取圖片
image_path = "animal.jpg"
image = cv2.imread(image_path)

# 轉換為 RGBA 格式（RemBG 需要）
image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

# 進行背景移除
result = remove(image_rgba)

# 儲存 PNG（保留透明背景）
output_path = "animal_no_bg.png"
cv2.imwrite(output_path, result)

# 顯示結果
cv2.imshow("Removed Background", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
