import cv2
import torch
import numpy as np
import os
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

# Tải mô hình một lần và giữ nó trong bộ nhớ
def load_model(opt):
    set_logging()
    device = select_device(opt['device'])
    half = device.type != 'cpu'
    model = attempt_load(opt['weights'], map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt['img-size'], s=stride)  # check img_size
    if half:
        model.half()
    return model, device, half, stride, imgsz

# Hàm để thực hiện suy luận
def run_inference(model, device, half, stride, imgsz, source_image_path, opt):
    img0 = cv2.imread(source_image_path)
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Get class names
    names = model.module.names if hasattr(model, 'module') else model.names
    

    # Generate colors for each class
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Inference
    with torch.no_grad():  # Disable gradient calculation for inference
        pred = model(img, augment=False)[0]

    # Apply NMS
    # classes = None
    # if opt['classes']:
    #     classes = []
    #     for class_name in opt['classes']:
    #         classes.append(opt['classes'].index(class_name))
    classes = [names.index("plasticubes")]  # Only keep 'plasticubes' class
    pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
    
    results = []  # List to store class names, confidence scores, and cropped images
    output_dir = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult"

    # for i, det in enumerate(pred):
    #     s = ''
    #     s += '%gx%g ' % img.shape[2:]  # print string
    #     #gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
    #     if len(det):
    #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

    #         for c in det[:, -1].unique():
    #             n = (det[:, -1] == c).sum()  # detections per class
    #             s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

    #         # for j, (*xyxy, conf, cls) in enumerate(reversed(det)):
    #         #     #label = f'{names[int(cls)]} {conf:.2f}'
    #         #     #plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

    #         #     # Append class name, confidence score, and cropped image to results
    #         #     xyxy = [int(x) for x in xyxy]
    #         #     cropped_img = img0[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    #         #     results.append((names[int(cls)], conf.item(), cropped_img))

    #         #     # Save the cropped image
    #         #     cv2.imwrite(os.path.join(output_dir, f'cropped_{os.path.basename(source_image_path).split(".")[0]}_{j}.bmp'), cropped_img)
    #         for j, (*xyxy, conf, cls) in enumerate(reversed(det)):
    #             if names[int(cls)] == "plasticubes":  # Chỉ xử lý class 'plasticubes'
    #                 # label = f'{names[int(cls)]} {conf:.2f}'
    #                 # plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

    #                 # Append class name, confidence score, and cropped image to results
    #                 xyxy = [int(x) for x in xyxy]
    #                 cropped_img = img0[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    #                 results.append((names[int(cls)], conf.item()))

    #                 # Lưu ảnh với tên dựa trên class
    #                 cv2.imwrite(
    #                     os.path.join(
    #                         output_dir,
    #                         f'cropped_{names[int(cls)]}_{os.path.basename(source_image_path).split(".")[0]}_{j}.bmp'
    #                     ),
    #                     cropped_img
    #                 )
    
    return results  # Return the list of class names, confidence scores, and cropped images

