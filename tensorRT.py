import torch
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

class EfficientNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        """Trích xuất feature maps từ các tầng cụ thể của EfficientNetV2-M."""
        super(EfficientNetFeatureExtractor, self).__init__()

        # Tải mô hình EfficientNetV2-M
        full_model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)

        # Chỉ giữ lại các tầng cần thiết: features[3], features[4], features[6]
        self.features = torch.nn.Sequential(
            full_model.features[3],
            full_model.features[4],
            torch.nn.Conv2d(160, 176, kernel_size=1, stride=1, padding=0),  # Điều chỉnh số kênh
            full_model.features[6],
        )

    def forward(self, x):
        """Truyền tensor qua các tầng cụ thể và thu thập feature maps."""
        feature_maps = []
        for layer in self.features:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps

# Bước 1: Tạo instance của lớp trích xuất
feature_extractor = EfficientNetFeatureExtractor().eval().cuda()

# Bước 2: Chuyển đổi sang TorchScript
scripted_extractor = torch.jit.script(feature_extractor)
torch.jit.save(scripted_extractor, "efficientnet_feature_extractor_scripted.pt")

# Bước 3: Tối ưu hóa với Torch-TensorRT
import torch_tensorrt

optimized_extractor = torch_tensorrt.compile(
    scripted_extractor,
    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions={torch.float},
    truncate_long_and_double=True
)

torch.jit.save(optimized_extractor, "efficientnet_feature_extractor_trt.pt")

# Bước 4: Sử dụng mô hình Torch-TensorRT
optimized_extractor = torch.jit.load("efficientnet_feature_extractor_trt.pt").eval().cuda()

# Tạo tensor đầu vào và truyền qua mô hình
input_tensor = torch.randn(1, 3, 224, 224).cuda()
with torch.no_grad():
    feature_maps = optimized_extractor(input_tensor)

# Resize các feature maps để đưa về cùng kích thước
resized_maps = [
    torch.nn.functional.adaptive_avg_pool2d(fmap, (28, 28)) for fmap in feature_maps
]

# Nối các feature maps
patch = torch.cat(resized_maps, 1)
patch = patch.reshape(patch.shape[1], -1).T

print("Kích thước feature maps sau khi xử lý:", patch.shape)
