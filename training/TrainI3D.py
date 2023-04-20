import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from i3d import InceptionI3d
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


INPUT_SIZE = (224, 224)
NUM_CLASSES = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.class_names = ["baby", "nature", "body", "father", "animal", "money", "family", "garden", "mother",
                            "morning", "fantastic", "happy", "hello", "number", "story", "remember", "please", "idea",
                            "sorry",
                            "stop"]
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for class_id, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_root, class_name)
            video_names = os.listdir(class_path)
            for video_name in video_names:
                samples.append({
                    "video_path": os.path.join(class_path, video_name),
                    "label": class_id
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]['video_path']
        label = self.samples[idx]['label']

        frames = self.load_frames(video_path)
        transformed_frames = [self.transform(frame) for frame in frames]

        inputs = torch.stack(transformed_frames, dim=0)
        inputs = inputs.view(3, 16, *INPUT_SIZE)  # Reshape the input tensor to (3, 16, height, width)
        return inputs, label

    def load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)

        cap.release()
        return frames


def remove_module_from_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    return new_state_dict


# Train function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        logits = outputs['logits']  # Extract the logits from the outputs dictionary
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 10 == 0:
            print(f"Training: Processed {i + 1} batches")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            logits = outputs['logits']  # Extract the logits from the outputs dictionary
            loss = criterion(logits, labels)

            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f"Evaluation: Processed {i + 1} batches")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc



train_data_root = "dataset/train"
val_data_root = "dataset/val"
test_data_root = "dataset/test"

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(train_data_root, transform=train_transforms)
val_dataset = CustomDataset(val_data_root, transform=val_test_transforms)
test_dataset = CustomDataset(test_data_root, transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, prefetch_factor=2)

pretrained_model = InceptionI3d(1064, in_channels=3)

state_dict = torch.load("/content/i3d.pth.tar")
state_dict = remove_module_from_keys(state_dict)
pretrained_model.load_state_dict(state_dict)

model = InceptionI3d(20, in_channels=3, num_in_frames=16)

for new_param, pretrained_param in zip(model.parameters(), pretrained_model.parameters()):
    if new_param.shape == pretrained_param.shape:
        new_param.data.copy_(pretrained_param.data)

# Set up the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
num_epochs = 25
model = model.to(device)

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    scheduler.step()

model_save_path = "13d_trained.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")