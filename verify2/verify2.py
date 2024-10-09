import torch
import os
from torchvision import transforms
import os
from PIL import Image


model = torch.load("model_path")
model.load_state_dict(torch.load("model_path"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
test_dir = "test_yellow_rust"
class_names = ["brown rust", "healthy", "septoria", "yellow rust"]
class_counts = {class_name: 0 for class_name in class_names}
model.eval()
for filename in os.listdir(test_dir):
  image_path = os.path.join(test_dir, filename)
  data_transforms = transforms.Compose([
      transforms.Lambda(lambda x: x.convert('RGB')),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  image = Image.open(image_path)
  image = data_transforms(image)
  image = image.unsqueeze(0)
  image = image.to(device)
  output = model(image)
  _, prediction = torch.max(output, 1)
  class_label = class_names[torch.argmax(prediction).item()]
  class_counts[class_label] += 1
  print(f"图片 {filename} 的分类结果为: {class_label}")
for class_name, count in class_counts.items():
  print(f"{class_name}: {count}")