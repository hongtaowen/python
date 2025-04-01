import os
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载预训练模型
    model = YOLO(r'D:\code pro1\pythonProject\gitbase\ultralytics\runs\segment\train17\weights\best.pt')

    # 设置输入图片文件夹路径
    input_folder = r"D:\code pro1\pythonProject\jiaxiang\QZjiaxiang\fanshejpg"

    # 获取文件夹中所有图片的路径，确保只读取jpg或png等特定格式的图片
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]

    # 批量推理并保存结果
    results = model(image_paths, nms=True, save=True, save_txt=True)

    # 处理结果
    for result in results:
        boxes = result.boxes  # 获取边界框输 出
        masks = result.masks  # 获取分割掩码输出
        keypoints = result.keypoints  # 获取关键点输出
        probs = result.probs  # 获取分类概率输出

    print("批量处理完成")
