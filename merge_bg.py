import cv2
import numpy as np

# 讀取去背後的圖片（透明 PNG）
foreground_path = "person_no_bg.png"
foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)  # 讀取 RGBA 格式

# 讀取新的背景圖片
background_path = "new_background.jpg"
background = cv2.imread(background_path)

# 確保背景尺寸適配
bg_h, bg_w = background.shape[:2]
fg_h, fg_w = foreground.shape[:2]

# 初始縮放比例
scale = 1.0
fg_w = int(fg_w * scale)
fg_h = int(fg_h * scale)

# 初始位置（讓人物置中）
x_offset = (bg_w - fg_w) // 2
y_offset = (bg_h - fg_h) // 2

# 滑鼠拖曳變數
dragging = False
start_x, start_y = 0, 0

# 滑鼠事件回調函數
def mouse_callback(event, x, y, flags, param):
    global x_offset, y_offset, start_x, start_y, dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        # 按下滑鼠左鍵開始拖曳
        dragging = True
        start_x, start_y = x - x_offset, y - y_offset

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            # 拖曳時更新位置
            x_offset = x - start_x
            y_offset = y - start_y

    elif event == cv2.EVENT_LBUTTONUP:
        # 放開滑鼠左鍵，停止拖曳
        dragging = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        # 滾輪縮放
        global scale, fg_w, fg_h, foreground
        if flags > 0:
            scale *= 1.1  # 放大
        else:
            scale /= 1.1  # 縮小

        # 限制縮放範圍
        scale = max(0.1, min(3.0, scale))

        # 重新計算尺寸
        fg_w = int(foreground.shape[1] * scale)
        fg_h = int(foreground.shape[0] * scale)

# 設定 OpenCV 視窗 & 滑鼠事件
cv2.namedWindow("Adjust Position")
cv2.setMouseCallback("Adjust Position", mouse_callback)

while True:
    # 建立新的畫布（背景）
    canvas = background.copy()

    # 確保圖片不超出範圍
    x_offset = min(max(x_offset, 0), bg_w - fg_w)
    y_offset = min(max(y_offset, 0), bg_h - fg_h)

    # 重新縮放去背後的人物
    fg_resized = cv2.resize(foreground, (fg_w, fg_h))

    # 分離前景的 RGB 與 Alpha 通道
    b, g, r, alpha = cv2.split(fg_resized)
    alpha = alpha / 255.0

    # 計算貼圖範圍
    y1, y2 = y_offset, y_offset + fg_h
    x1, x2 = x_offset, x_offset + fg_w

    # 確保範圍不超過背景
    if x2 > bg_w or y2 > bg_h:
        continue

    # 遮罩處理，將前景疊加到背景
    for c in range(3):
        canvas[y1:y2, x1:x2, c] = (foreground[y1-y_offset:y2-y_offset, x1-x_offset:x2-x_offset, c] * alpha +
                                    background[y1:y2, x1:x2, c] * (1 - alpha)).astype(np.uint8)

    # 顯示畫面
    cv2.imshow("Adjust Position", canvas)

    # 按下 "Enter" 鍵（或 ESC）確認位置並退出
    key = cv2.waitKey(1) & 0xFF
    if key == 13 or key == 27:
        break

# 儲存最終圖片
cv2.imwrite("person_with_new_bg.png", canvas)
cv2.destroyAllWindows()
