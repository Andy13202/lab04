import cv2
import numpy as np

# 讀取圖片
image_path = "person.jpg"  # 請更換為你的圖片
image = cv2.imread(image_path)

# 取得圖片尺寸
height, width = image.shape[:2]

# 讓學生輸入參數
print("請輸入去背參數（按 Enter 使用預設值）：")

blur_amount = input("1️⃣ 模糊程度（0-10，影響邊緣平滑度，預設 3）：")
blur_amount = int(blur_amount) if blur_amount.isdigit() else 3
blur_amount = max(0, min(blur_amount, 10))  # 限制範圍 0-10

fg_threshold = input("2️⃣ 前景閥值（50-255，影響保留範圍，預設 150）：")
fg_threshold = int(fg_threshold) if fg_threshold.isdigit() else 150
fg_threshold = max(50, min(fg_threshold, 255))  # 限制範圍 50-255

morph_size = input("3️⃣ 形態學處理（0-5，影響邊緣細節，預設 2）：")
morph_size = int(morph_size) if morph_size.isdigit() else 2
morph_size = max(0, min(morph_size, 5))  # 限制範圍 0-5

# 建立遮罩
mask = np.zeros((height, width), np.uint8)

# 建立背景與前景模型（GrabCut 需要的格式）
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# 設定 ROI（感興趣區域），假設人物位於圖像中央
rect = (10, 10, width - 20, height - 20)

# 應用 GrabCut
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# 生成前景遮罩
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

# 應用形態學處理來優化遮罩
kernel = np.ones((morph_size, morph_size), np.uint8)
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)  # 閉運算填補小孔洞

# 應用遮罩
result = image * mask2[:, :, np.newaxis]

# 應用模糊來平滑邊緣
if blur_amount > 0:
    result = cv2.GaussianBlur(result, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)

# 轉換為 4 通道（RGBA），將背景變為透明
b, g, r = cv2.split(result)
alpha = mask2 * fg_threshold  # 根據前景閥值調整透明度
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
