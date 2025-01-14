import time
import cv2
import torch
import numpy as np
import os
#from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
#from utils.plots import plot_one_box
from utils.torch_utils import select_device

from pathlib import Path
import os, shutil
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchvision.transforms import transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from torchvision.models import resnet50, ResNet50_Weights


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
    print(f"Using device: {device}")  # Thêm thông báo để xác nhận thiết bị đang sử dụng
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
    classes = None
    if opt['classes']:
        classes = []
        for class_name in opt['classes']:
            classes.append(opt['classes'].index(class_name))

    pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
    results = []
    output_dir2 = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult"
    for i, det in enumerate(pred):
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        #gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            for j, (*xyxy, conf, cls) in enumerate(reversed(det)):
                if names[int(cls)] == "plasticubes":  # Chỉ xử lý class 'plasticubes'
                    # label = f'{names[int(cls)]} {conf:.2f}'
                    # plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                    # Append class name, confidence score, and cropped image to results
                    xyxy = [int(x) for x in xyxy]
                    cropped_img = img0[xyxy[1]-10:xyxy[3]+10, xyxy[0]-10:xyxy[2]+10]
                    results.append((names[int(cls)], conf.item()))
                
                    # cv2.imwrite(output_path, img)
                    # print(f"Đã lưu ảnh đã xử lý: {output_path}")
                    # gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 0.8 , 30, param1=35, param2=0.3, minRadius=60, maxRadius=250)
                    
                    # if circles is not None:
                    #     circles = np.uint16(np.around(circles))
                    #     count = 0
                    #     radius = []
                    #     for i in circles[0, :]:
                    #         #cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 255), 1)
                    #         #cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
                    #         #print(i[2])
                    #         radius.append(i[2])
                    #         count += 1
                    #     print("  ")
                    # else:
                    #     pass
                    # #Kiểm tra và tăng bán kính nếu count = 1
                    # if count == 1:
                    #     mask = np.zeros_like(gray)
                    #     #cv2.circle(mask, (i[0], i[1]), i[2]+14, 255, thickness=-1)  # White background
                    #     cv2.circle(mask, (i[0], i[1]), i[2]+7, 255, thickness=-1) # Black background
                    #     # Lấy các pixel ngoài đường tròn (phần còn lại sẽ là 0)
                    #     mask_inv = cv2.bitwise_not(mask)
                    #     # Tô màu trắng cho các pixel ngoài viền đường tròn
                    #     #img[mask_inv == 255] = (255, 255, 255) # White background
                    #     cropped_img[mask_inv == 255] = (0, 0, 0) # Black background
                    #     #print("1")
                    #     radius.clear()
                    # elif count == 2:
                    #     if radius[0] > radius[1]:
                    #         mask = np.zeros_like(gray)
                    #         #cv2.circle(mask, (i[0], i[1]), radius[0]+9, 255, thickness=-1) # White background
                    #         cv2.circle(mask, (i[0], i[1]), radius[0]+3, 255, thickness=-1) # Black background
                    #         mask_inv = cv2.bitwise_not(mask)
                    #         #img[mask_inv == 255] = (255, 255, 255)
                    #         cropped_img[mask_inv == 255] = (0, 0, 0) # Black background
                    #         #print("2")
                    #         radius.clear()
                    #     else:
                    #         mask = np.zeros_like(gray)
                    #         #cv2.circle(mask, (i[0], i[1]), radius[1]+9, 255, thickness=-1) # White background
                    #         cv2.circle(mask, (i[0], i[1]), radius[1]+3, 255, thickness=-1) # Black background
                    #         mask_inv = cv2.bitwise_not(mask)
                    #         #img[mask_inv == 255] = (255, 255, 255) # White background
                    #         cropped_img[mask_inv == 255] = (0, 0, 0)
                    #         radius.clear()
                    # elif count >= 3:
                    #     max = radius[0] 
                    #     for r in radius[1:]:
                    #         if r >= max:
                    #             max = r
                    #     mask = np.zeros_like(gray)
                    #     #cv2.circle(mask, (i[0], i[1]), int(max)+9, 255, thickness=-1) # White background
                    #     cv2.circle(mask, (i[0], i[1]), int(max)+3, 255, thickness=-1) # Black background
                    #     mask_inv = cv2.bitwise_not(mask)
                    #     #img[mask_inv == 255] = (255, 255, 255) # White background
                    #     cropped_img[mask_inv == 255] = (0, 0, 0) # Black background
                    #     #print("3")
                    #     radius.clear()
                    # else:
                    #     pass
                    img = cv2.resize(cropped_img, (224,224))
                    #img = cv2.blur(cropped_img, (5, 5))
                    img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                    ptsright = np.array([[216, 0], [216, 224], [224, 224], [224, 0]], np.int32)
                    ptsright = ptsright.reshape((-1, 1, 2))
                    cv2.fillPoly(img, [ptsright], (0, 0, 0))
                    ptstop = np.array([[0, 0], [0, 11], [224, 11], [224, 0]], np.int32)
                    ptstop = ptstop.reshape((-1, 1, 2))
                    cv2.fillPoly(img, [ptstop], (0, 0, 0))
                    ptsbottom = np.array([[0, 217], [0, 224], [224, 224], [224, 217]], np.int32)
                    ptsbottom = ptsbottom.reshape((-1, 1, 2))
                    cv2.fillPoly(img, [ptsbottom], (0, 0, 0))
                    ptsleft = np.array([[0, 0], [0, 224], [4, 224], [4, 0]], np.int32)
                    ptsleft = ptsleft.reshape((-1, 1, 2))
                    cv2.fillPoly(img, [ptsleft], (0, 0, 0))
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


                    #cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
                    #cropped_img_resize = cv2.resize(cropped_img, (224,224))
                    # Lưu ảnh với tên dựa trên class
                    cv2.imwrite(
                        os.path.join(
                            output_dir2,
                            f'{names[int(cls)]}_{os.path.basename(source_image_path).split(".")[0]}_{j}.png'
                        ),
                        img
                    )
                # elif names[int(cls)] == "foamtrays":  # Chỉ xử lý class 'plasticubes'
                #     xyxy = [int(x) for x in xyxy]
                #     cropped_img = img0[xyxy[1]-20:xyxy[3]+20, xyxy[0]-20:xyxy[2]+20]
                #     results.append((names[int(cls)], conf.item()))

                #     cv2.imwrite(
                #         os.path.join(
                #             output_dir2,
                #             f'{names[int(cls)]}_{os.path.basename(source_image_path).split(".")[0]}_{j}.bmp'
                #         ),
                #         cropped_img
                #     )

    return results

def setup_ad():
    transform = transforms.Compose([ transforms.Resize((224,224)), transforms.ToTensor() ])
    
    # class efficientnet_feature_extractor(torch.nn.Module):
    #     def __init__(self):
    #         """This class extracts the feature maps from a pretrained EfficientNetV2-M model."""
    #         super(efficientnet_feature_extractor, self).__init__()
    #         # Tải mô hình EfficientNetV2-M với trọng số đã huấn luyện
    #         self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)

    #         # Đặt chế độ đánh giá và tắt gradient
    #         self.model.eval()
    #         for param in self.model.parameters():
    #             param.requires_grad = False

    #         # Hook để trích xuất feature maps
    #         def hook(module, input, output):
    #             """Lưu lại các feature maps từ các tầng cụ thể."""
    #             self.features.append(output)

    #         self.model.features[3].register_forward_hook(hook)
    #         self.model.features[4].register_forward_hook(hook)  # Tầng giữa (middle layer)
    #         self.model.features[6].register_forward_hook(hook)  # Tầng sâu hơn (deeper layer)

    #     def forward(self, input):
    #         """Truyền đầu vào qua mô hình và thu thập feature maps."""
    #         self.features = []  # Để lưu các feature maps
    #         with torch.no_grad():
    #             _ = self.model(input)  # Truyền qua mô hình

    #         # Feature maps từ các hook
    #         resized_maps = [
    #             torch.nn.functional.adaptive_avg_pool2d(fmap, (28, 28)) for fmap in self.features
    #         ]
    #         patch = torch.cat(resized_maps, 1)  # Nối các feature maps
    #         patch = patch.reshape(patch.shape[1], -1).T  # Chuyển thành tensor dạng cột

    #         return patch
    class resnet_feature_extractor(torch.nn.Module):
        def __init__(self):
            """This class extracts the feature maps from a pretrained Resnet model."""
            super(resnet_feature_extractor, self).__init__()
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            # Hook to extract feature maps
            def hook(module, input, output) -> None:
                """This hook saves the extracted feature map on self.featured."""
                self.features.append(output)

            self.model.layer2[-1].register_forward_hook(hook)
            self.model.layer3[-1].register_forward_hook(hook)

        def forward(self, input):

            self.features = []
            with torch.no_grad():
                _ = self.model(input)

            self.avg = torch.nn.AvgPool2d(3, stride=1)
            fmap_size = self.features[0].shape[-2]         # Feature map sizes h, w
            self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)

            resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
            patch = torch.cat(resized_maps, 1)            # Merge the resized feature maps
            patch = patch.reshape(patch.shape[1], -1).T   # Craete a column tensor

            return patch

    #backbone = efficientnet_feature_extractor().cuda()
    backbone = resnet_feature_extractor().cuda()
    memory_bank1 = torch.load(r"H:\HK241\NCKH\Model\memory_bank_resnet_blue_V1\memory_bank_resnet_blue_V1.pt").cuda()
    memory_bank2 = torch.load(r"H:\HK241\NCKH\Model\efficientnet_v2_m\memory_bank2.pt").cuda()

    return backbone, memory_bank1, memory_bank2, transform




    #cv2.imshow('Result', img0)
    #cv2.imwrite(f'Result_{os.path.basename(image_path)}', img0)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
if __name__ == '__main__':
    # Thiết lập các tham số và đường dẫn
    classes_to_filter = None  # You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person']
    source_folder_path = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\SourceImages"
    opt = {
        "weights": r"H:\HK241\NCKH\Model\cubesdetection.pt",  # Path to weights file default weights are for nano model
        "img-size": 480,  # default image size
        "conf-thres": 0.8,  # confidence threshold for inference.
        "iou-thres": 0.3,  # NMS IoU threshold for inference.
        "device": '0' if torch.cuda.is_available() else 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
        "classes": classes_to_filter  # list of classes to filter or None
    }
    t1 = time.time()
    # Tải mô hình một lần
    model, device, half, stride, imgsz = load_model(opt)
    backbone, memory_bank1, memory_bank2, transform =  setup_ad()
    
    t2 = time.time()
    print(f'Load Done. ({t2 - t1:.3f}s)')
    # Lấy danh sách các ảnh trong thư mục
    image_paths = [os.path.join(source_folder_path, f) for f in os.listdir(source_folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    t3 = time.time()
    # # Thực hiện suy luận cho từng ảnh trong thư mục
    for image_path in image_paths:
        results = run_inference(model, device, half, stride, imgsz, image_path, opt)
        for class_name, confidence in results:
            print(f"Class: {class_name}, Confidence: {confidence:.2f}")
    t4 = time.time()
    print(f'Model. ({t4 - t3:.3f}s)')
    print(f'Check. ({t4 - t2:.3f}s)')
    print(f'Done. ({t4 - t1:.3f}s)')
    # input_folder = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult"
    # output_folder = r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\AnomalyBlueTest"
    # for filename in os.listdir(input_folder):
    #     if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
    #         input_path = os.path.join(input_folder, filename)
    #         output_path = os.path.join(output_folder, f"{filename}")

    #         img = cv2.imread(input_path)
    #         if img is None:
    #             print(f"Không thể đọc ảnh: {input_path}")
    #             continue

    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 0.8 , 30, param1=35, param2=0.3, minRadius=60, maxRadius=250)
            
    #         if circles is not None:
    #             circles = np.uint16(np.around(circles))
    #             count = 0
    #             radius = []
    #             for i in circles[0, :]:
    #                 #cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 255), 1)
    #                 #cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    #                 #print(i[2])
    #                 radius.append(i[2])
    #                 count += 1
    #             print("  ")
    #         else:
    #             print(f"Không tìm thấy đường tròn trong ảnh: {input_path}")
    #         #Kiểm tra và tăng bán kính nếu count = 1
    #         if count == 1:
    #             mask = np.zeros_like(gray)
    #             #cv2.circle(mask, (i[0], i[1]), i[2]+14, 255, thickness=-1)  # White background
    #             #cv2.circle(mask, (i[0], i[1]), i[2]+7, 255, thickness=-1) # Black background
    #             cv2.circle(mask, (i[0], i[1]), i[2], 255, thickness=-1)
    #             # Lấy các pixel ngoài đường tròn (phần còn lại sẽ là 0)
    #             mask_inv = cv2.bitwise_not(mask)
    #             # Tô màu trắng cho các pixel ngoài viền đường tròn
    #             #img[mask_inv == 255] = (255, 255, 255) # White background
    #             img[mask_inv == 255] = (0, 0, 0) # Black background
    #             #print("1")
    #             radius.clear()
    #         elif count == 2:
    #             if radius[0] > radius[1]:
    #                 mask = np.zeros_like(gray)
    #                 #cv2.circle(mask, (i[0], i[1]), radius[0]+9, 255, thickness=-1) # White background
    #                 #cv2.circle(mask, (i[0], i[1]), radius[0]+3, 255, thickness=-1) # Black background
    #                 cv2.circle(mask, (i[0], i[1]), radius[0], 255, thickness=-1)
    #                 mask_inv = cv2.bitwise_not(mask)
    #                 #img[mask_inv == 255] = (255, 255, 255)
    #                 img[mask_inv == 255] = (0, 0, 0) # Black background
    #                 #print("2")
    #                 radius.clear()
    #             else:
    #                 mask = np.zeros_like(gray)
    #                 #cv2.circle(mask, (i[0], i[1]), radius[1]+9, 255, thickness=-1) # White background
    #                 #cv2.circle(mask, (i[0], i[1]), radius[1]+3, 255, thickness=-1) # Black background
    #                 cv2.circle(mask, (i[0], i[1]), radius[0], 255, thickness=-1)
    #                 mask_inv = cv2.bitwise_not(mask)
    #                 #img[mask_inv == 255] = (255, 255, 255) # White background
    #                 img[mask_inv == 255] = (0, 0, 0)
    #                 radius.clear()
    #         elif count >= 3:
    #             max = radius[0] 
    #             for r in radius[1:]:
    #                 if r >= max:
    #                     max = r
    #             mask = np.zeros_like(gray)
    #             #cv2.circle(mask, (i[0], i[1]), int(max)+9, 255, thickness=-1) # White background
    #             #cv2.circle(mask, (i[0], i[1]), int(max)+3, 255, thickness=-1) # Black background
    #             cv2.circle(mask, (i[0], i[1]), int(max), 255, thickness=-1)
    #             mask_inv = cv2.bitwise_not(mask)
    #             #img[mask_inv == 255] = (255, 255, 255) # White background
    #             img[mask_inv == 255] = (0, 0, 0) # Black background
    #             #print("3")
    #             radius.clear()
    #         else:
    #             print(f"Không tìm thấy đường tròn trong ảnh: {input_path}")
    #             break  # Thoát vòng lặp nếu không tìm thấy đường tròn
    #         img = cv2.resize(img, (224,224))
    #         #img = cv2.blur(img, (5, 5))
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         ptsright = np.array([[216, 0], [216, 224], [224, 224], [224, 0]], np.int32)
    #         ptsright = ptsright.reshape((-1, 1, 2))
    #         cv2.fillPoly(img, [ptsright], (0, 0, 0))
    #         ptstop = np.array([[0, 0], [0, 11], [224, 11], [224, 0]], np.int32)
    #         ptstop = ptstop.reshape((-1, 1, 2))
    #         cv2.fillPoly(img, [ptstop], (0, 0, 0))
    #         ptsbottom = np.array([[0, 217], [0, 224], [224, 224], [224, 217]], np.int32)
    #         ptsbottom = ptsbottom.reshape((-1, 1, 2))
    #         cv2.fillPoly(img, [ptsbottom], (0, 0, 0))
    #         ptsleft = np.array([[0, 0], [0, 224], [4, 224], [4, 0]], np.int32)
    #         ptsleft = ptsleft.reshape((-1, 1, 2))
    #         cv2.fillPoly(img, [ptsleft], (0, 0, 0))
    #         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
    #         cv2.imwrite(output_path, img)
      
            # print(f"Đã lưu ảnh đã xử lý: {output_path}")
            

    # # #Đảm bảo backbone ở chế độ đánh giá
    # backbone.eval()

    # # Đường dẫn đến thư mục cần kiểm tra
    # test_path = Path(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult")
    # results_counter = Counter()

    # # Lặp qua tất cả các tệp .bmp trong thư mục
    # for path in test_path.glob('*.png'):  # Tìm các tệp .bmp trong thư mục 'bad'
    #     t6 = time.time()
    #     test_image = transform(Image.open(path)).cuda().unsqueeze(0)

    #     # Dự đoán với mô hình
    #     with torch.no_grad():
    #         features = backbone(test_image)

    #     # Tính toán khoảng cách
    #     distances = torch.cdist(features, memory_bank1, p=2.0)
    #     dist_score, dist_score_idxs = torch.min(distances, dim=1)
    #     s_star = torch.max(dist_score)

    #     # Tính điểm bất thường
    #     y_score_image = s_star.cpu().numpy()
    #     y_pred_image = 1 * (y_score_image >= 12.104041100)
    #     class_label = ['GOOD', 'BAD']
    #     results_counter[class_label[y_pred_image]] += 1
    #     # In kết quả
    #     print(f'File name: {path.name}')
    #     print(f'Anomaly score: {y_score_image:0.9f}')
    #     print(f'Prediction: {class_label[y_pred_image]}')
    #     # Load the image
    #     # image = cv2.imread(str(path))
    #     # text = class_label[y_pred_image]
    #     # font = cv2.FONT_HERSHEY_SIMPLEX
    #     # cv2.putText(image, text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    #     # # Display the image using OpenCV
    #     # cv2.imshow('Image', image)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     t7 = time.time()
    #     print(f'Once Processing. ({t7 - t6:.3f}s)')
    # t5 = time.time()
    # print(f'Processing Finished. ({t5 - t2:.3f}s)')

    # print(f"GOOD: {results_counter['GOOD']}")
    # print(f"BAD: {results_counter['BAD']}")
    # #xoa tat ca cac file trong thu muc ImagesResult và addgood
    # #shutil.rmtree(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult")
    # #shutil.rmtree(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\addgood")
    # #os.makedirs(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult")
    # #os.makedirs(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\addgood")
    # print("Các tệp đã được xóa thành công.")



