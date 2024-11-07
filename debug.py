import torchvision

# Lấy danh sách các lớp từ bộ dữ liệu ImageNet
classes = torchvision.datasets.ImageNet.classes
print(classes)