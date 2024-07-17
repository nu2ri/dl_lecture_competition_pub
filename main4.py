import re, random, time, os
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
from transformers import DebertaV2TokenizerFast, DebertaV2Model
from accelerate import Accelerator
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
tokenizer = DebertaV2TokenizerFast.from_pretrained('microsoft/deberta-v2-xlarge')
deberta_model = DebertaV2Model.from_pretrained('microsoft/deberta-v2-xlarge')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def most_confident_answer(answers):
    answers = [answer['answer'] for answer in answers]
    common_answer = Counter(answers).most_common(1)[0][0]
    return common_answer

# 1. データローダーの作成
class VQADataset(Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.df = pd.read_json(df_path)
        self.image_dir = image_dir
        self.transform = transform
        self.answer = answer

        if self.answer:
            self.df['most_confident_answer'] = self.df['answers'].apply(most_confident_answer)
            self.df['most_confident_answer'] = self.df['most_confident_answer'].apply(process_text)
            self.df['most_confident_answer_idx'] = self.df['most_confident_answer'].factorize()[0]
            self.answer2idx = {a: idx for idx, a in enumerate(self.df['most_confident_answer'].unique())}
            self.idx2answer = {idx: a for a, idx in self.answer2idx.items()}

        self.question2idx = {q: idx for idx, q in enumerate(self.df['question'].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, row['image'])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        question = process_text(row['question'])
        inputs = tokenizer(question, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        question_embedding = deberta_model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)

        if self.answer:
            answer_idx = row['most_confident_answer_idx']
            return image, question_embedding, answer_idx
        else:
            return image, question_embedding

    def update_dict(self, train_dataset):
        self.question2idx = train_dataset.question2idx
        if self.answer:
            self.answer2idx = train_dataset.answer2idx
            self.idx2answer = train_dataset.idx2answer



# 2. 評価指標の実装
# 簡単にするならBCEを利用する
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


# 3. モデルのの実装
# ResNetを利用できるようにしておく
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
        self.fc2 = nn.Linear(512 + 1536, n_answer)

    def forward(self, image, question_embedding):
        image_feature = self.resnet(image)  # 画像の特徴量
        image_feature = self.fc1(image_feature)
        x = torch.cat((image_feature, question_embedding), dim=1)
        output = self.fc2(x)
        return output


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    simple_acc = 0
    start = time.time()
    for images, questions, answers in tqdm(dataloader):
        images, questions, answers = images.to(device), questions.to(device), answers.to(device)
        optimizer.zero_grad()
        with accelerator.autocast():
            outputs = model(images, questions)
            loss = criterion(outputs, answers)
        accelerator.backward(loss)
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == answers).sum().item()
        try:
            simple_acc += (preds == mode(answers.tolist())).sum().item()
        except StatisticsError:
            pass

    return total_loss / len(dataloader), correct / len(dataloader.dataset), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    simple_acc = 0
    start = time.time()
    for images, questions, answers in dataloader:
        images, questions, answers = images.to(device), questions.to(device), answers.to(device)
        with accelerator.autocast():
            outputs = model(images, questions)
            loss = criterion(outputs, answers)
        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == answers).sum().item()
        simple_acc += (preds == mode(answers).values[0]).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset), simple_acc / len(dataloader), time.time() - start


def main():
    # deviceの設定
    set_seed(3407)
    accelerator = Accelerator()
    device = accelerator.device

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)
    # optimizer / criterion
    optimizer = RAdam(model.parameters(), lr=1e-3, weight_decay=1e-5, weight_decouple=True, adam_debias=True)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    num_epoch = 10

    # train model
    for epoch in tqdm(range(num_epoch)):
        model.train()
        total_loss = 0.0
        correct = 0
        simple_acc = 0
        start = time.time()
        for images, questions, answers in train_loader:
            images, questions, answers = images.to(device), questions.to(device), answers.to(device)
            optimizer.zero_grad()
            with accelerator.autocast():
                outputs = model(images, questions)
                loss = criterion(outputs, answers)
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == answers).sum().item()
            try:
                simple_acc += (preds == mode(answers.tolist())).sum().item()
            except StatisticsError:
                pass
        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)
        train_simple_acc = simple_acc / len(train_loader)
        train_time = time.time() - start
        # train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        with torch.no_grad():
            pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer.get(id, 'unknown') for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model3.pth")
    np.save("submission3.npy", submission)

if __name__ == "__main__":
    main()
