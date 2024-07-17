import re, random, time, os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from statistics import mode, StatisticsError

from PIL import Image
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_optimizer import RAdam
from tqdm import tqdm
from transformers import AlbertTokenizer, AlbertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    text = text.lower()
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)
    text = re.sub(r"[^\w\s':]", ' ', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def most_confident_answer(answers):
    answers = [answer['answer'] for answer in answers]
    common_answer = Counter(answers).most_common(1)[0][0]
    return common_answer

class ZCAWhitening(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(ZCAWhitening, self).__init__()
        self.epsilon = epsilon
        self.mean = None
        self.zca_matrix = None

    def fit(self, x):
        N, C, H, W = x.shape
        x = x.view(N, -1)
        
        self.mean = x.mean(dim=0, keepdim=True)
        x = x - self.mean

        sigma = torch.mm(x.T, x) / N
        U, S, V = torch.svd(sigma)
        
        self.zca_matrix = torch.mm(torch.mm(U, torch.diag(1.0/torch.sqrt(S + self.epsilon))), U.T)

    def forward(self, x):
        if self.mean is None or self.zca_matrix is None:
            raise RuntimeError("ZCAWhitening transform must be fit before it can be applied.")
        
        x = x.view(-1).unsqueeze(0)
        x = x - self.mean
        x = torch.mm(x, self.zca_matrix.T)
        return x.view(x.shape[1:])

    def to(self, device):
        super().to(device)
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.zca_matrix is not None:
            self.zca_matrix = self.zca_matrix.to(device)
        return self

class VQADataset(Dataset):
    def __init__(self, df_path, image_dir, answer=True, preload=True):
        self.df = pd.read_json(df_path)
        self.image_dir = image_dir
        self.answer = answer
        self.preload = preload
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')
        self.pretrained_model = AlbertModel.from_pretrained('albert-large-v2').to(device)
        self.pretrained_model.eval()

        self.to_tensor = transforms.ToTensor()
        self.preload_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.gpu_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.zca_whitening = ZCAWhitening()

        if self.answer:
            self.df['most_confident_answer'] = self.df['answers'].apply(most_confident_answer)
            self.df['most_confident_answer'] = self.df['most_confident_answer'].apply(process_text)
            self.df['most_confident_answer_idx'] = self.df['most_confident_answer'].factorize()[0]
            self.answer2idx = {a: idx for idx, a in enumerate(self.df['most_confident_answer'].unique())}
            self.idx2answer = {idx: a for a, idx in self.answer2idx.items()}
        self.question2idx = {q: idx for idx, q in enumerate(self.df['question'].unique())}

        if self.preload:
            self.preload_images()

    def preload_images(self):
        print("Preloading images into GPU memory...")
        self.images = []
        try:
            for i in tqdm(range(len(self.df))):
                image_path = os.path.join(self.image_dir, self.df.iloc[i]['image'])
                image = Image.open(image_path).convert("RGB")
                image = self.preload_transform(image).to(device)
                self.images.append(image)
            print("Images preloaded successfully.")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU out of memory during preloading. Falling back to on-the-fly loading.")
                self.preload = False
                self.images = None
                torch.cuda.empty_cache()
            else:
                raise e

    def fit_zca_whitening(self, batch_size=2):
        print("Fitting ZCA Whitening...")
        if self.preload:
            images = torch.stack(self.images)
            self.zca_whitening.fit(images)
        else:
            temp_loader = DataLoader(
                self,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=lambda x: [item[0] for item in x]
            )
            n_samples = 0
            mean_sum = None
            cov_sum = None
            for batch in tqdm(temp_loader, desc="Computing ZCA parameters"):
                batch = torch.stack(batch).to(device)
                batch_size = batch.size(0)
                n_samples += batch_size
                batch = batch.view(batch_size, -1)
                if mean_sum is None:
                    mean_sum = torch.sum(batch, dim=0)
                else:
                    mean_sum += torch.sum(batch, dim=0)
                if cov_sum is None:
                    cov_sum = torch.mm(batch.t(), batch)
                else:
                    cov_sum += torch.mm(batch.t(), batch)
                torch.cuda.empty_cache()
            self.zca_whitening.mean = (mean_sum / n_samples).unsqueeze(0)
            sigma = (cov_sum / n_samples) - torch.mm(self.zca_whitening.mean.t(), self.zca_whitening.mean)
            U, S, V = torch.svd(sigma)
            self.zca_whitening.zca_matrix = torch.mm(
                torch.mm(U, torch.diag(1.0 / torch.sqrt(S + self.zca_whitening.epsilon))),
                U.t()
            )
        self.zca_whitening = self.zca_whitening.to(device)
        print("ZCA Whitening fitted.")

    def update_dict(self, dataset):
        self.question2idx = dataset.question2idx
        if self.answer:
            self.answer2idx = dataset.answer2idx
            self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.preload:
            image = self.images[idx]
        else:
            image = Image.open(os.path.join(self.image_dir, row['image'])).convert("RGB")
            image = self.preload_transform(image).to(device)
        if self.zca_whitening.mean is not None and self.zca_whitening.zca_matrix is not None:
            image = self.zca_whitening(image)
        image = self.gpu_transform(image)
        question = process_text(row['question'])
        inputs = self.tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            question_embedding = self.pretrained_model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
        if self.answer:
            answer_idx = torch.tensor(row['most_confident_answer_idx'], device=device)
            return image, question_embedding, answer_idx
        else:
            return image, question_embedding

    def __len__(self):
        return len(self.df)

def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.
    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10
    return total_acc / len(batch_pred)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])

class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int):
        super(VQAModel, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512 + 1024, n_answer)

    def forward(self, image, question_embedding):
        image_feature = self.resnet(image)
        image_feature = self.fc1(image_feature)
        x = torch.cat((image_feature, question_embedding), dim=1)
        output = self.fc2(x)
        return output

def train(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    correct = 0
    simple_acc = 0
    start = time.time()
    for images, questions, answers in tqdm(dataloader):
        images, questions, answers = images.to(device), questions.to(device), answers.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images, questions)
            loss = criterion(outputs, answers)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == answers).sum().item()
        simple_acc += (preds == mode(answers.tolist())).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    simple_acc = 0
    start = time.time()
    with torch.no_grad():
        for images, questions, answers in tqdm(dataloader):
            images, questions, answers = images.to(device), questions.to(device), answers.to(device)
            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == answers).sum().item()
            simple_acc += (preds == mode(answers).values[0]).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset), simple_acc / len(dataloader), time.time() - start

def main():
    set_seed(3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", preload=True)
    # train_dataset.fit_zca_whitening()
    batch_size = 32 if train_dataset.preload else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", answer=False, preload=False)
    test_dataset.update_dict(train_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)

    num_epoch = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = RAdam(model.parameters(), lr=1e-3, weight_decay=1e-5, weight_decouple=True, adam_debias=True)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm(range(num_epoch)):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device, scaler)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer.get(id, 'unknown') for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model3.pth")
    np.save("submission3.npy", submission)

if __name__ == "__main__":
    main()
