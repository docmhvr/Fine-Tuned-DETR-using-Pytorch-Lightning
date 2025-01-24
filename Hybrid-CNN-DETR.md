import torch
import torchvision.transforms as T
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader
from roboflow import Roboflow
import os

# Step 1: Roboflow dataset setup
# Sign up at Roboflow (https://roboflow.com) and get the API key
import os
api_key = os.getenv("ROBOFLOW_API_KEY")  # Use environment variable for API key
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY environment variable is not set")

rf = Roboflow(api_key=api_key)
project = rf.workspace().project("your_project_name")
dataset = project.version(1).download("coco")  # Download COCO format dataset

# Step 2: Define dataset loader class
from torchvision.datasets import CocoDetection

class PowerlineDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        if self.transform:
            img = self.transform(img)
        return img, target

# Step 3: Load pre-trained DETR model
model_name = "facebook/detr-resnet-50"
model = DetrForObjectDetection.from_pretrained(model_name)
processor = DetrImageProcessor.from_pretrained(model_name)

# Step 4: Define transformations
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Step 5: Create DataLoader
data_dir = dataset.location
train_dataset = PowerlineDataset(
    root=os.path.join(data_dir, "train"),
    annFile=os.path.join(data_dir, "train", "_annotations.coco.json"),
    transform=transform
)

val_dataset = PowerlineDataset(
    root=os.path.join(data_dir, "valid"),
    annFile=os.path.join(data_dir, "valid", "_annotations.coco.json"),
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: x)

# Step 6: Fine-tuning the DETR model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for imgs, targets in train_loader:
        pixel_values = processor(images=imgs, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        # Prepare labels
        labels = [{"class_labels": [ann["category_id"] for ann in target["annotations"]]} for target in targets]

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("detr_powerline_model")
processor.save_pretrained("detr_powerline_model")

# Step 7: Evaluate on validation set
model.eval()
for imgs, targets in val_loader:
    with torch.no_grad():
        pixel_values = processor(images=imgs, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        outputs = model(pixel_values=pixel_values)

    print(outputs.logits.argmax(-1))  # Predicted class labels
