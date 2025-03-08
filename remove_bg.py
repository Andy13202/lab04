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

# 創建 OpenCV 介面
cv2.namedWindow("Settings")

# 初始化參數
def update(_):
    # 取得滑桿的數值
    iterations = cv2.getTrackbarPos("Iterations", "Settings") + 1
    fg_threshold = cv2.getTrackbarPos("FG Threshold", "Settings")
    bg_threshold = cv2.getTrackbarPos("BG Threshold", "Settings")

    # 重新建立 GrabCut 遮罩
    mask[:] = 0
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    # 生成前景遮罩（根據閥值調整）
    mask2 = np.where((mask == 2) & (mask >= bg_threshold) | (mask == 0), 0, 1).astype("uint8")
    
    # 應用遮罩
    result = image * mask2[:, :, np.newaxis]

    # 轉換為 RGBA，背景變透明
    b, g, r = cv2.split(result)
    alpha = mask2 * 255
    result_rgba = cv2.merge([b, g, r, alpha])

    # 顯示調整結果
    cv2.imshow("Removed Background", result_rgba)

# 建立滑桿（讓學生調整）
cv2.createTrackbar("Iterations", "Settings", 1, 10, update)   # 影響 GrabCut 運行次數
cv2.createTrackbar("FG Threshold", "Settings", 1, 3, update)  # 影響前景偵測
cv2.createTrackbar("BG Threshold", "Settings", 0, 2, update)  # 影響背景偵測

# 初次運行
update(0)

# 按下 Enter 儲存結果
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 13 or key == 27:  # Enter 或 ESC 退出
        break

# 最終儲存圖片
output_path = "person_no_bg.png"
cv2.imwrite(output_path, cv2.imread("Removed Background"))

cv2.destroyAllWindows()
