#TalChernovich
#original + weighted loss
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import datetime
import csv
import numpy as np

# ==== 1. הגדר נתיבי דאטא ====
train_dir = r"D:\DataFinal\trainDIRnoduplicates"
val_dir = r"D:\DataFinal\valDIRnoduplicates"
batch_size = 16
num_epochs = 30
lr = 1e-4

# ==== 2. הגדר שם לתיקיית התוצאות והמודל ====
run_name = "originalnoduplicates+diffweight([5.0 ,3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 5.0, 5.0] 30ep syntheticsaar6ai)"
output_dir = os.path.join("F:/testsyntetic/models/postsyntetic", run_name)
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, f"{run_name}.pt")
history_path = os.path.join(output_dir, f"history_{run_name}.csv")
graph_path = os.path.join(output_dir, f"training_curves_{run_name}.png")

# ==== 3. הגדר טרנספורמציות ואוגמנטציה ====
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==== 4. טען דאטאסט ====
train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
num_classes = len(train_ds.classes)
print("מחלקות:", train_ds.classes)

# ==== 5. הגדר מודל ====
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ==== 6. קריטריון ואופטימייזר (הוספתי שקלול מחלקות ידני) ====

# עדכן את המשקלים כרצונך - הסדר לפי train_ds.classes
class_weights = torch.tensor([5.0 ,3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 5.0, 5.0], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

# ==== 7. משתנים לשמירת נתונים ====
history = []
best_acc = 0.0

# ==== 8. לולאת אימון ====
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    # אימות
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    print(f"אפוק {epoch+1}/{num_epochs} - train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}")

    # שמור נתונים להיסטוריה
    history.append([epoch+1, train_loss, train_acc, val_loss, val_acc])

    # שמור את המודל הטוב ביותר
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path)

    scheduler.step(val_loss)

print(f"אימון הסתיים! המודל הכי טוב נשמר ב־{model_path}")

# ==== 9. שמירת גרף ו־CSV ====
with open(history_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    writer.writerows(history)

# גרף
epochs = [row[0] for row in history]
train_losses = [row[1] for row in history]
val_losses = [row[3] for row in history]
train_accs = [row[2] for row in history]
val_accs = [row[4] for row in history]

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")

plt.tight_layout()
plt.savefig(graph_path)
plt.close()

print(f"שמרתי את הגרפים ל־{graph_path}")
print(f"שמרתי את ההיסטוריה ל־{history_path}")

# ==== 10. Compute and plot confusion matrix ====
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_ds.classes)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Validation Set")
plt.savefig(os.path.join(output_dir, f"confusion_matrix_{run_name}.png"))
plt.show()
