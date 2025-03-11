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
print("2️⃣ 手動繪製遮罩（需要滑鼠操作）")
init_method = input("請輸入選項 1 或 2（預設 1）：")
init_method = int(init_method) if init_method.isdigit() and int(init_method) in [1, 2] else 1

# 讓學生輸入參數
blur_amount = input("1️⃣ 邊緣平滑度（影響邊緣柔和度，0-10，預設 3）：")
blur_amount = int(blur_amount) if blur_amount.isdigit() else 3
blur_amount = max(0, min(blur_amount, 10))

morph_size = input("2️⃣ 形態學處理（影響邊緣細節，0-5，預設 2）：")
morph_size = int(morph_size) if morph_size.isdigit() else 2
morph_size = max(0, min(morph_size, 5))

# 建立遮罩
mask = np.zeros((height, width), np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

if init_method == 1:
    # 自動矩形選取
    rect = (10, 10, width - 20, height - 20)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

else:
    # 讓學生手動繪製前景區域
    print("🖌 請在圖片上 **繪製前景區域**，然後關閉視窗來繼續。")

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    foreground_mask = np.zeros((height, width), np.uint8)

    points = []  # 存放學生標記的點

    def onclick(event):
        """滑鼠點擊事件，用來標記前景區域"""
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            points.append((x, y))
            ax.scatter(x, y, c='r', s=10)  # 標記點
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(points) > 0:
        # 生成前景遮罩（基於點的區域）
        points = np.array([points], np.int32)
        cv2.fillPoly(foreground_mask, points, 255)

        # GrabCut 初始化
        mask[foreground_mask == 255] = cv2.GC_FGD
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    else:
        print("⚠️ 未標記前景，使用預設矩形選取。")
        rect = (10, 10, width - 20, height - 20)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# 生成前景遮罩
mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")

# 形態學處理
kernel = np.ones((morph_size, morph_size), np.uint8)
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

# 應用遮罩
result = image * mask2[:, :, np.newaxis]

# 邊緣平滑處理
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
