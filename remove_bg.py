import cv2
import numpy as np

# 讀取圖片
image_path = "person.jpg"  # 請更換為你的圖片
image = cv2.imread(image_path)

# 取得圖片尺寸
height, width = image.shape[:2]

# 讓學生輸入參數
print("請輸入去背參數（按 Enter 使用預設值）：")

iterations = input("1️⃣ 迭代次數（推薦範圍 1-10，預設 5）：")
iterations = int(iterations) if iterations.isdigit() else 5

fg_threshold = input("2️⃣ 前景強度（推薦範圍 1-3，預設 2）：")
fg_threshold = int(fg_threshold) if fg_threshold.isdigit() else 2

bg_threshold = input("3️⃣ 背景強度（推薦範圍 0-2，預設 1）：")
bg_threshold = int(bg_threshold) if bg_threshold.isdigit() else 1

# 建立遮罩
mask = np.zeros((height, width), np.uint8)

# 建立背景與前景模型（GrabCut 需要的格式）
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# 設定 ROI（感興趣區域），假設人物位於圖像中央
rect = (10, 10, width - 20, height - 20)

# 應用 GrabCut（基於矩形區域進行初步前景分割）
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

# 生成前景遮罩（根據閥值調整）
mask2 = np.where((mask == 2) & (mask >= bg_threshold) | (mask == 0), 0, 1).astype("uint8")

# 應用遮罩
result = image * mask2[:, :, np.newaxis]

# 轉換為 4 通道（RGBA），將背景變為透明
b, g, r = cv2.split(result)
alpha = mask2 * 255
result_rgba = cv2.merge([b, g, r, alpha])

# 儲存為 PNG（支援透明背景）
output_path = "person_no_bg.png"
cv2.imwrite(output_path, result_rgba)

# 顯示結果
cv2.imshow("Original", image)
cv2.imshow("Removed Background", result_rgba)

print(f"✅ 去背完成！已儲存為 {output_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()
