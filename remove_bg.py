import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖片
image_path = input("請輸入圖片路徑（例如：person.jpg）：").strip()
image = cv2.imread(image_path)

if image is None:
    print("❌ 圖片讀取失敗，請確認檔案路徑是否正確！")
    exit()

# 取得圖片尺寸
height, width = image.shape[:2]

# 讓學生選擇 GrabCut 的初始化方式
print("請選擇 GrabCut 初始化方式：")
print("1️⃣ 自動矩形選取區域（預設）")
print("2️⃣ 手動選擇區域（需要滑鼠操作，若無法顯示 GUI，會改用 `matplotlib`）")
init_method = input("請輸入選項 1 或 2（預設 1）：")
init_method = int(init_method) if init_method.isdigit() and int(init_method) in [1, 2] else 1

# 讓學生輸入參數
blur_amount = input("1️⃣ 邊緣平滑度（影響邊緣柔和度，0-10，預設 3）：")
blur_amount = int(blur_amount) if blur_amount.isdigit() else 3
blur_amount = max(0, min(blur_amount, 10))  # 限制範圍 0-10

morph_size = input("2️⃣ 形態學處理（影響邊緣細節，0-5，預設 2）：")
morph_size = int(morph_size) if morph_size.isdigit() else 2
morph_size = max(0, min(morph_size, 5))  # 限制範圍 0-5

mask_threshold = input("3️⃣ 遮罩閥值（影響前景保留，0-3，預設 1）：")
mask_threshold = int(mask_threshold) if mask_threshold.isdigit() else 1
mask_threshold = max(0, min(mask_threshold, 3))  # 限制範圍 0-3

# 建立遮罩
mask = np.zeros((height, width), np.uint8)

# 建立背景與前景模型（GrabCut 需要的格式）
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

if init_method == 1:
    # 自動矩形選取
    rect = (10, 10, width - 20, height - 20)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
else:
    try:
        cv2.imshow("請選擇 ROI", image)
        cv2.waitKey(100)  # 給 OpenCV GUI 一點時間啟動
        roi = cv2.selectROI("請選擇 ROI", image, False, False)
        cv2.destroyAllWindows()
        rect = (roi[0], roi[1], roi[2], roi[3])
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except:
        print("⚠️ 無法使用 OpenCV `selectROI`，改用 `matplotlib` 來選擇區域。")
        def get_roi_manually(image):
            fig, ax = plt.subplots()
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            roi = plt.ginput(2)  # 讓使用者點擊兩個角落來選擇 ROI
            plt.close(fig)
            return roi

        roi_points = get_roi_manually(image)
        x1, y1 = map(int, roi_points[0])
        x2, y2 = map(int, roi_points[1])
        rect = (x1, y1, x2 - x1, y2 - y1)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# 生成前景遮罩
mask2 = np.where((mask == mask_threshold) | (mask == 0), 0, 1).astype("uint8")

# **形態學處理**
kernel = np.ones((morph_size, morph_size), np.uint8)
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)  # 填補小孔洞
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)  # 去除小雜訊

# 應用遮罩
result = image * mask2[:, :, np.newaxis]

# **邊緣平滑處理**
if blur_amount > 0:
    result = cv2.GaussianBlur(result, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)

# 轉換為 4 通道（RGBA），將背景變為透明
b, g, r = cv2.split(result)
alpha = mask2 * 255  # 設定透明度
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
