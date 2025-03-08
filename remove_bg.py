import cv2
import numpy as np

def nothing(x):
    pass

# 读取输入图像
image_path = 'input.jpg'  # 请替换为你的图片路径
img = cv2.imread(image_path)

# 创建窗口和滑动条
cv2.namedWindow('Adjust Parameters')

# 颜色范围参数（适用于 HSV 空间）
cv2.createTrackbar('H_min', 'Adjust Parameters', 0, 180, nothing)
cv2.createTrackbar('H_max', 'Adjust Parameters', 180, 180, nothing)
cv2.createTrackbar('S_min', 'Adjust Parameters', 0, 255, nothing)
cv2.createTrackbar('S_max', 'Adjust Parameters', 255, 255, nothing)
cv2.createTrackbar('V_min', 'Adjust Parameters', 0, 255, nothing)
cv2.createTrackbar('V_max', 'Adjust Parameters', 255, 255, nothing)

# GrabCut 迭代次数
cv2.createTrackbar('GrabCut Iter', 'Adjust Parameters', 1, 10, nothing)

# 处理图像
while True:
    # 获取滑动条的值
    h_min = cv2.getTrackbarPos('H_min', 'Adjust Parameters')
    h_max = cv2.getTrackbarPos('H_max', 'Adjust Parameters')
    s_min = cv2.getTrackbarPos('S_min', 'Adjust Parameters')
    s_max = cv2.getTrackbarPos('S_max', 'Adjust Parameters')
    v_min = cv2.getTrackbarPos('V_min', 'Adjust Parameters')
    v_max = cv2.getTrackbarPos('V_max', 'Adjust Parameters')
    grabcut_iter = cv2.getTrackbarPos('GrabCut Iter', 'Adjust Parameters')

    # 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 颜色阈值分割
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 反转掩码：保留人物部分
    mask_inv = cv2.bitwise_not(mask)

    # 运行 GrabCut 进行进一步分割
    mask_gc = np.zeros(img.shape[:2], np.uint8)
    mask_gc[mask_inv == 255] = cv2.GC_FGD  # 设为前景
    mask_gc[mask_inv == 0] = cv2.GC_BGD  # 设为背景

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 应用 GrabCut 进行图像分割
    cv2.grabCut(img, mask_gc, None, bgd_model, fgd_model, grabcut_iter, cv2.GC_INIT_WITH_MASK)

    # 处理输出
    final_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)
    result = cv2.bitwise_and(img, img, mask=final_mask)

    # 显示结果
    cv2.imshow('Result', result)

    # 按 's' 保存图片，按 'q' 退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite('segmented_output.png', result)
        print("去背图片已保存： segmented_output.png")
    elif key == ord('q'):
        break

# 关闭窗口
cv2.destroyAllWindows()