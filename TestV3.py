import torch
from pathlib import Path
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from torchvision.transforms import transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from PIL import Image
import time
from concurrent.futures import ThreadPoolExecutor
import os
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
    class efficientnet_feature_extractor(torch.nn.Module):
        def __init__(self):
            """Extract feature maps from a pretrained EfficientNetV2-M model."""
            super(efficientnet_feature_extractor, self).__init__()
            self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)

            # Đặt chế độ đánh giá và tắt gradient
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        def forward(self, input):
            """Trích xuất các feature maps trực tiếp."""
            # Truyền qua các tầng và thu thập feature maps
            x = input
            features = []
            for i, layer in enumerate(self.model.features):
                x = layer(x)
                if i in [3, 4, 6]:  # Chỉ lấy feature maps từ các tầng cụ thể
                    features.append(x)

            # Resize và gộp các feature maps
            resized_maps = [
                torch.nn.functional.adaptive_avg_pool2d(fmap, (28, 28)) for fmap in features
            ]
            patch = torch.cat(resized_maps, 1)  # Nối các feature maps
            patch = patch.reshape(patch.shape[1], -1).T  # Chuyển thành tensor dạng cột

            return patch
    backbone = efficientnet_feature_extractor().cuda()
    scripted_model = torch.jit.script(backbone)
    scripted_model.save(r"H:\HK241\NCKH\Model\efficientnet_v2_m\efficientnet_feature_extractor_scripted.pt")
    memory_bank1 = torch.load(r"H:\HK241\NCKH\Model\efficientnet_v2_m\memory_bank1_2.pt").cuda()
    memory_bank2 = torch.load(r"H:\HK241\NCKH\Model\efficientnet_v2_m\memory_bank2_2.pt").cuda()
    scripted_model = torch.jit.load(r"H:\HK241\NCKH\Model\efficientnet_v2_m\efficientnet_feature_extractor_scripted.pt")
    return backbone, memory_bank1, memory_bank2, transform, scripted_model

def process_image(path, transform, scripted_model, memory_bank1):
    try:
        t1 = time.time()
        test_image = transform(Image.open(path)).cuda().unsqueeze(0)

        # Dự đoán với mô hình
        with torch.no_grad():
            features = scripted_model(test_image)

        # Tính toán khoảng cách
        distances = torch.cdist(features, memory_bank1, p=2.0)
        dist_score, _ = torch.min(distances, dim=1)
        s_star = torch.max(dist_score)

        # Tính điểm bất thường
        y_score_image = float(s_star)
        y_pred_image = 1 * (y_score_image >= 102.21337890625)
        class_label = ['GOOD', 'BAD']

        # Kết quả
        return path.name, class_label[y_pred_image], y_score_image

    except Exception as e:
        return path.name, "ERROR", str(e)


if __name__ == '__main__':

    backbone, memory_bank1, memory_bank2, transform, scripted_model =  setup_ad()
    backbone.eval()
    t1 = time.time()
    test_path = Path(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult")
    image_paths = list(test_path.glob('*.jpg'))

    # Số lượng CPU cores khả dụng
    cpu_count = os.cpu_count()  # Hoặc multiprocessing.cpu_count()
    print(f"Số lượng CPU cores khả dụng: {cpu_count}")
    # Sử dụng ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_image, path, transform, backbone, memory_bank1) for path in image_paths]
        for future in futures:
            results.append(future.result())
    
    # In kết quả
    for result in results:
        print(result)
    t2 = time.time()
    print(f'Processing Finished. ({t2 - t1:.3f}s)')
    #scripted_model.eval()
    # Đường dẫn đến thư mục cần kiểm tra
    # test_path = Path(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult")

    # # Lặp qua tất cả các tệp .bmp trong thư mục
    # for path in test_path.glob('*.bmp'):  # Tìm các tệp .bmp trong thư mục 'bad'
    #     t1 = time.time()
    #     test_image = transform(Image.open(path)).cuda().unsqueeze(0)

    #     # Dự đoán với mô hình
    #     t3 = time.time()
    #     with torch.no_grad():
    #         #with autocast():
    #         features = scripted_model(test_image)
    #     t4 = time.time()

    #     # Tính toán khoảng cách
    #     distances = torch.cdist(features, memory_bank1, p=2.0)
    #     dist_score, dist_score_idxs = torch.min(distances, dim=1)
    #     s_star = torch.max(dist_score)
    #     #print(f"s_star: {s_star.item():.4f}")
    #     t5 = time.time()

    #     # Tính điểm bất thường
    #     #y_score_image = s_star.cpu().numpy()
    #     y_score_image = float(s_star)
    #     y_pred_image = 1 * (y_score_image >= 102.21337890625)
    #     class_label = ['GOOD', 'BAD']
    #     t6 = time.time()

    #     # In kết quả
    #     print(f'File name: {path.name}')
    #     print(f'Anomaly score: {y_score_image:0.4f}')
    #     print(f'Prediction: {class_label[y_pred_image]}')
    #     t2 = time.time()
        
    #     print(f'Inference. ({t4 - t3:.5f}s)')
    #     print(f'Cal. ({t5 - t4:.5f}s)')
    #     print(f'Predict. ({t6 - t5:.5f}s)')
    #     print(f'Once Processing. ({t2 - t1:.5f}s)')

 
