import cv2
import numpy as np

# 讀取去背後的 PNG 圖片（透明背景）
foreground_path = "person_no_bg.png"
foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)  # 讀取 RGBA

# 讀取新背景圖片
background_path = "new_background.jpg"
background = cv2.imread(background_path)

# 確保背景大小
bg_h, bg_w = background.shape[:2]
fg_h, fg_w = foreground.shape[:2]

# 初始縮放比例
scale = 1.0
fg_w = int(fg_w * scale)
fg_h = int(fg_h * scale)

# 初始位置（置中）
x_offset = (bg_w - fg_w) // 2
y_offset = (bg_h - fg_h) // 2

# 滑鼠拖曳變數
dragging = False
start_x, start_y = 0, 0

# 滑鼠事件回調函數
def mouse_callback(event, x, y, flags, param):
    global x_offset, y_offset, start_x, start_y, dragging, scale, fg_w, fg_h, foreground

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_x, start_y = x - x_offset, y - y_offset

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            x_offset = x - start_x
            y_offset = y - start_y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scale *= 1.1  # 放大
        else:
            scale /= 1.1  # 縮小
        scale = max(0.1, min(3.0, scale))

        fg_w = int(foreground.shape[1] * scale)
        fg_h = int(foreground.shape[0] * scale)

cv2.namedWindow("Adjust Position")
cv2.setMouseCallback("Adjust Position", mouse_callback)

while True:
    # 建立背景副本
    canvas = background.copy()

    # 確保不超出邊界
    x_offset = min(max(x_offset, 0), bg_w - fg_w)
    y_offset = min(max(y_offset, 0), bg_h - fg_h)

    # 計算貼圖範圍（避免超出背景）
    y1, y2 = y_offset, min(y_offset + fg_h, bg_h)
    x1, x2 = x_offset, min(x_offset + fg_w, bg_w)

    # 計算實際的貼圖大小
    fg_h_new = y2 - y1
    fg_w_new = x2 - x1

    # 縮放人物圖像，使其與背景對齊
    fg_resized = cv2.resize(foreground, (fg_w_new, fg_h_new))

    # 分離前景的 RGB 與 Alpha 通道
    b, g, r, alpha = cv2.split(fg_resized)

    # 確保 alpha 遮罩大小正確
    alpha = alpha.astype(float) / 255.0
    alpha = alpha[:, :, np.newaxis]  # (h, w) → (h, w, 1)

    # **修正：確保 `alpha` 和 `background` 的形狀匹配**
    alpha = np.concatenate([alpha] * 3, axis=2)  # 轉換為 (h, w, 3)，與 RGB 通道匹配

    # 創建 RGB 前景圖像
    foreground_rgb = cv2.merge([b, g, r])

    # **修正：確保 `foreground_rgb` 和 `background[y1:y2, x1:x2]` 形狀相同**
    foreground_rgb = cv2.resize(foreground_rgb, (fg_w_new, fg_h_new))

    # **修正：確保 `background[y1:y2, x1:x2]` 的形狀與前景匹配**
    background_patch = background[y1:y2, x1:x2]
    background_patch = cv2.resize(background_patch, (fg_w_new, fg_h_new))

    # **混合前景與背景**
    blended = (foreground_rgb * alpha + background_patch * (1 - alpha)).astype(np.uint8)

    # 放回背景
    canvas[y1:y2, x1:x2] = blended

    # 顯示畫面
    cv2.imshow("Adjust Position", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == 13 or key == 27:
        break

# 儲存結果
cv2.imwrite("person_with_new_bg.png", canvas)
cv2.destroyAllWindows()
