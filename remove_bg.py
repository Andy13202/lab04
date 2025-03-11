import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖片
image_path = "person.jpg"  # 請更換為你的圖片
image = cv2.imread(image_path)

# 取得圖片尺寸
height, width = image.shape[:2]

# 讓學生輸入參數
print("請輸入去背參數（不輸入的話，直接按 Enter 可使用預設值）：")

iterations = input("1️⃣ GrabCut 迭代次數（影響去背精細度，1-10，預設 5）：")
iterations = int(iterations) if iterations.isdigit() else 5
iterations = max(1, min(iterations, 10))

blur_amount = input("2️⃣ 邊緣平滑度（0-10，影響邊緣平滑度，預設 3）：")
blur_amount = int(blur_amount) if blur_amount.isdigit() else 3
blur_amount = max(0, min(blur_amount, 10))

morph_size = input("3️⃣ 形態學處理（0-10，影響邊緣細節，預設 5）：")
morph_size = int(morph_size) if morph_size.isdigit() else 5
morph_size = max(0, min(morph_size, 10))

# 建立遮罩（所有像素預設為背景）
mask = np.zeros((height, width), np.uint8)

# 建立 GrabCut 背景與前景模型
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# 設定 ROI（假設人物在圖像中央）
rect = (10, 10, width - 20, height - 20)

# **可視化初始遮罩**
plt.subplot(2, 3, 1)
plt.imshow(mask, cmap="gray")
plt.title("Initial Mask (All Background)")
plt.axis("off")

# **執行 GrabCut**
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

# **GrabCut 運行後的遮罩**
mask_visual = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype("uint8")

plt.subplot(2, 3, 2)
plt.imshow(mask_visual, cmap="gray")
plt.title("Mask After GrabCut")
plt.axis("off")

# **應用形態學處理**
kernel = np.ones((morph_size, morph_size), np.uint8)

# **閉運算（填補小孔洞）**
mask_closed = cv2.morphologyEx(mask_visual, cv2.MORPH_CLOSE, kernel)

# **開運算（去除小噪點）**
mask_opened = cv2.morphologyEx(mask_visual, cv2.MORPH_OPEN, kernel)

# **邊緣梯度運算（讓學生看到邊緣變化）**
mask_gradient = cv2.morphologyEx(mask_visual, cv2.MORPH_GRADIENT, kernel)

plt.subplot(2, 3, 3)
plt.imshow(mask_closed, cmap="gray")
plt.title("Morphology Close (Fill Holes)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(mask_opened, cmap="gray")
plt.title("Morphology Open (Remove Noise)")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(mask_gradient, cmap="gray")
plt.title("Edge Gradient (Show Edge Changes)")
plt.axis("off")

plt.show()

# **應用閉運算處理後的遮罩**
mask_final = mask_closed

# **應用遮罩**
result = image * mask_final[:, :, np.newaxis]

# **應用邊緣平滑**
if blur_amount > 0:
    result = cv2.GaussianBlur(result, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)

# **轉換為 4 通道（RGBA），讓背景透明**
b, g, r = cv2.split(result)
alpha = mask_final * 255  # 設定透明度
result_rgba = cv2.merge([b, g, r, alpha])

# **儲存最終結果**
output_path = "person_no_bg.png"
cv2.imwrite(output_path, result_rgba)

# **顯示最終去背結果**
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("After")
plt.axis("off")

plt.show()

print(f"✅ 去背完成！已儲存為 {output_path}")
