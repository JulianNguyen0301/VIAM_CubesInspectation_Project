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
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.optim as optim
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def setup_ad():
    transform = transforms.Compose([ transforms.Resize((224,224)), transforms.ToTensor() ])
    
    class efficientnet_feature_extractor(torch.nn.Module):
        def __init__(self):
            """This class extracts the feature maps from a pretrained EfficientNetV2-M model."""
            super(efficientnet_feature_extractor, self).__init__()
            # Tải mô hình EfficientNetV2-M với trọng số đã huấn luyện
            self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)

            # Đặt chế độ đánh giá và tắt gradient
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            # Hook để trích xuất feature maps
            def hook(module, input, output):
                """Lưu lại các feature maps từ các tầng cụ thể."""
                self.features.append(output)

            self.model.features[3].register_forward_hook(hook)
            self.model.features[4].register_forward_hook(hook)  # Tầng giữa (middle layer)
            self.model.features[6].register_forward_hook(hook)  # Tầng sâu hơn (deeper layer)

        def forward(self, input):
            """Truyền đầu vào qua mô hình và thu thập feature maps."""
            self.features = []  # Để lưu các feature maps
            with torch.no_grad():
                _ = self.model(input)  # Truyền qua mô hình

            # Feature maps từ các hook
            resized_maps = [
                torch.nn.functional.adaptive_avg_pool2d(fmap, (28, 28)) for fmap in self.features
            ]
            patch = torch.cat(resized_maps, 1)  # Nối các feature maps
            patch = patch.reshape(patch.shape[1], -1).T  # Chuyển thành tensor dạng cột

            return patch
    backbone = efficientnet_feature_extractor().cuda()
    memory_bank1 = torch.load(r"H:\HK241\NCKH\Model\efficientnet_v2_m\memory_bank1_2.pt").cuda()
    memory_bank2 = torch.load(r"H:\HK241\NCKH\Model\efficientnet_v2_m\memory_bank2_2.pt").cuda()

    return backbone, memory_bank1, memory_bank2, transform

if __name__ == '__main__':
    
    backbone, memory_bank1, memory_bank2, transform =  setup_ad()
    
#     t2 = time.time()
    

#     # Đảm bảo backbone ở chế độ đánh giá
#     backbone.eval()

#     # Đường dẫn đến thư mục cần kiểm tra
#     test_path = Path(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult")

#     # Lặp qua tất cả các tệp .bmp trong thư mục
#     for path in test_path.glob('*.bmp'):  # Tìm các tệp .bmp trong thư mục 'bad'
#         test_image = transform(Image.open(path)).cuda().unsqueeze(0)

#         # Dự đoán với mô hình
#         with torch.no_grad():
#             features = backbone(test_image)

#         # Tính toán khoảng cách
#         distances = torch.cdist(features, memory_bank1, p=2.0)
#         dist_score, dist_score_idxs = torch.min(distances, dim=1)
#         s_star = torch.max(dist_score)

#         # Tính điểm bất thường
#         y_score_image = s_star.cpu().numpy()
#         y_pred_image = 1 * (y_score_image >= 102.21337890625)
#         class_label = ['GOOD', 'BAD']

#         # In kết quả
#         print(f'File name: {path.name}')
#         print(f'Anomaly score: {y_score_image:0.4f}')
#         print(f'Prediction: {class_label[y_pred_image]}')
#     t5 = time.time()
#     print(f'Processing Finished. ({t5 - t2:.3f}s)')

    # Đảm bảo backbone ở chế độ đánh giá
# Đảm bảo backbone ở chế độ đánh giá
backbone.eval()

# Hàm xử lý từng ảnh
def process_image(path, transform, backbone, memory_bank1, class_label):
    try:
        # Chuẩn bị dữ liệu
        test_image = transform(Image.open(path)).cuda().unsqueeze(0)

        # Dự đoán với mô hình
        with torch.no_grad():
            features = backbone(test_image)

        # Tính toán khoảng cách
        distances = torch.cdist(features, memory_bank1, p=2.0)
        dist_score, _ = torch.min(distances, dim=1)
        s_star = torch.max(dist_score)

        # Tính điểm bất thường và dự đoán
        y_score_image = s_star.cpu().numpy()
        y_pred_image = 1 * (y_score_image >= 102.21337890625)
        
        # Kết quả cho ảnh hiện tại
        return {
            'file_name': path.name,
            'anomaly_score': y_score_image,
            'prediction': class_label[y_pred_image]
        }
    except Exception as e:
        return {'file_name': path.name, 'error': str(e)}

# Pipeline chính
if __name__ == "__main__":
    # Đường dẫn đến thư mục cần kiểm tra
    test_path = Path(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult")
    image_paths = list(test_path.glob('*.bmp'))  # Tìm tất cả các tệp .bmp trong thư mục

    # Cấu hình
    class_label = ['GOOD', 'BAD']

    # Xử lý đa luồng
    t_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:  # Sử dụng 10 luồng
        futures = [executor.submit(process_image, path, transform, backbone, memory_bank1, class_label) for path in image_paths]
        for future in futures:
            results.append(future.result())

    # In tất cả kết quả
    for result in results:
        if 'error' in result:
            print(f"Error processing {result['file_name']}: {result['error']}")
        else:
            print(f"File name: {result['file_name']}")
            print(f"Anomaly score: {result['anomaly_score']:.4f}")
            print(f"Prediction: {result['prediction']}")

    t_end = time.time()
    print(f'Processing Finished. Total time: {t_end - t_start:.3f}s')