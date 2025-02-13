import cv2
import numpy as np

def main():
    # 用户输入图像路径
    image_path = input("请输入图像路径: ")
    image = cv2.imread(image_path)
    if image is None:
        print("图像加载失败，请检查路径。")
        return

    # 创建一个与输入图像同样大小的掩码，并用可能的背景标记初始化
    mask = np.zeros(image.shape[:2], np.uint8)

    # 定义背景和前景模型（内部使用）
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 定义包含前景的矩形区域（格式为：x, y, w, h）
    # 注意：这个矩形应尽量紧凑地包围前景对象
    rect = (50, 50, image.shape[1]-100, image.shape[0]-100)  # 根据实际情况调整

    # 运行 GrabCut 算法
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # 将背景区域的标记转换为0，前景为1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 使用掩码提取前景
    image_fg = image * mask2[:, :, np.newaxis]

    # 显示和保存结果
    cv2.imshow("Original", image)
    cv2.imshow("GrabCut Mask", mask2)
    cv2.imshow("Foreground", image_fg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果图像
    cv2.imwrite("foreground_extracted.png", image_fg)

if __name__ == "__main__":
    main()