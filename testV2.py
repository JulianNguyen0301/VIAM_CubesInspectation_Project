import torch
from pathlib import Path
from PIL import Image
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
import time
from torch.amp import autocast
import torch_tensorrt
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
    # backbone1 = efficientnet_feature_extractor().eval().cuda()
    # inputs = [torch.randn((1, 3, 224, 224)).cuda()] # define a list of representative inputs here

    # trt_gm = torch_tensorrt.compile(backbone1, ir="dynamo", inputs=inputs)
    # torch_tensorrt.save(trt_gm, "trt.ep", inputs=inputs) # PyTorch only supports Python runtime for an ExportedProgram. For C++ deployment, use a TorchScript file
    # torch_tensorrt.save(trt_gm, "trt.ts", output_format="torchscript", inputs=inputs)
    

    return backbone, memory_bank1, memory_bank2, transform

if __name__ == '__main__':
    # t1 = time.time()
    # backbone, memory_bank1, memory_bank2, transform =  setup_ad()
    # t2 = time.time()
    # print(f'Once Processing. ({t2 - t1:.5f}s)')
    # scripted_model = torch.jit.load(r"H:\HK241\NCKH\Model\efficientnet_v2_m\efficientnet_feature_extractor_scripted.pt")
    # #print(scripted_model)
    # scripted_model.eval()
    # image = transform(Image.open(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult\processed_cropped_IMG_0023_15.bmp"))
    # image = image.unsqueeze(0).cuda()


    backbone, memory_bank1, memory_bank2, transform =  setup_ad()
    backbone.eval()
    scripted_model = torch.jit.load(r"H:\HK241\NCKH\Model\efficientnet_v2_m\efficientnet_feature_extractor_scripted.pt")
    scripted_model.eval()
    # Đường dẫn đến thư mục cần kiểm tra
    test_path = Path(r"H:\APP UNIVERSITY\CODE PYTHON\CVWembley\ImagesResult")

    # Lặp qua tất cả các tệp .bmp trong thư mục
    for path in test_path.glob('*.jpg'):  # Tìm các tệp .bmp trong thư mục 'bad'
        t1 = time.time()
        test_image = transform(Image.open(path)).cuda().unsqueeze(0)

        # Dự đoán với mô hình
        t3 = time.time()
        with torch.no_grad():
            #with autocast():
            features = scripted_model(test_image)
        t4 = time.time()
        # Tính toán khoảng cách
        distances = torch.cdist(features, memory_bank1, p=2.0)
        dist_score, dist_score_idxs = torch.min(distances, dim=1)
        s_star = torch.max(dist_score)
        t5 = time.time()

        # Tính điểm bất thường
        #y_score_image = s_star.cpu().numpy()
        y_score_image = float(s_star)
        y_pred_image = 1 * (y_score_image >= 102.21337890625)
        class_label = ['GOOD', 'BAD']
        t6 = time.time()

        # In kết quả
        print(f'File name: {path.name}')
        print(f'Anomaly score: {y_score_image:0.4f}')
        print(f'Prediction: {class_label[y_pred_image]}')
        t2 = time.time()
        
        print(f'Inference. ({t4 - t3:.5f}s)')
        print(f'Cal. ({t5 - t4:.5f}s)')
        print(f'Predict. ({t6 - t5:.5f}s)')
        print(f'Once Processing. ({t2 - t1:.5f}s)')

 
