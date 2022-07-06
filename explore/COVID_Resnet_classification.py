import random
import os 
import cv2
import matplotlib.pyplot as plt
from pathlib import Path 

os.chdir("/Users/kimjh/Documents/Bio_DeepLearning_summer/")

# data path -> and image check 
covid_train_path = os.path.join('data/Covid_Dataset', 'Covid_Train', '1_CT_COVID')

covid_files = [os.path.join(covid_train_path, x) for x in os.listdir(covid_train_path)]
covid_images =  [cv2.imread(x) for x in random.sample(covid_files, 5)]

# plt.figure(figsize=(20,10))
# columns = 5
# for i, image in enumerate(covid_images):
#     plt.subplot(len(covid_images) / columns + 1, columns, i + 1)
#     plt.imshow(image)


# data proportion check 
root_dir = "/Users/kimjh/Documents/Bio_DeepLearning_summer/data/Covid_Dataset/"

def get_files(root_dir, prefix):
    files = os.listdir(root_dir)
    lab = ['COVID', 'Non-COVID']
    for i, file in enumerate(files):
        path = os.path.join(root_dir, file)
        file_list = os.listdir(path)
        print(f"{prefix} 데이터의 {lab[i]} 수: {len(file_list)}")

phase = "Train"
get_files(root_dir + f"Covid_{phase}", phase)
print()
phase = "Valid"
get_files(root_dir + f"Covid_{phase}", phase)
print()
phase = "Test"
get_files(root_dir + f"Covid_{phase}", phase)

# Data preprocessing
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    # torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.deterministic = True 

seed_everything(42)

# transform 적용 
image_transform = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224), scale = (0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0,0,0], std = [1,1,1]),
    ]),

    'valid': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0,0,0], std = [1,1,1]),
    ])
}

train_dataset_folder = f'{root_dir}Covid_Train'
val_dataset_folder = f'{root_dir}Covid_Valid'
test_dataset_folder = f'{root_dir}Covid_Test'

train_dataset = datasets.ImageFolder(root=train_dataset_folder, transform=image_transform['train'])
valid_dataset = datasets.ImageFolder(root=val_dataset_folder, transform=image_transform['valid'])
test_dataset = datasets.ImageFolder(root=test_dataset_folder, transform=image_transform['valid'])

batch_size = 64 
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, pin_memory=True, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=True, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True, pin_memory=True, drop_last=False)

for x, y in train_loader:
    print(x.shape)
    print(y.shape)
    break


# get model 
from torchvision import models 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 

config = {
    # Classifier 설정 
    'cls_hidden_dims' : []
}

class CovidResNet(nn.Module):
    """ pretrained model 활용 """
    def __init__(self):
        """
        Args: 
            base_model: resnet18 / resnet50
            config: 모델 설정 값 
        """
        super(CovidResNet, self).__init__()
        model = models.resnet50(pretrained=True) # from torchvision import models -> 통해서 pretrained model get
        num_ftrs = model.fc.in_features # fully connected layer channel 수 얻기 
        self.num_ftrs = num_ftrs 

        for name, param in model.named_parameters(): # named_paramters() = name, parameters를 반환  
            if 'layer2' in name:
                break
            param.requires_grad = False
        
        self.features = nn.Sequential(*list(model.children())[:-1])
    
    def forward(self,x):
        x = self.features(x)
        b = x.size(0)
        x = x.view(b, -1)
        return x 

model_image = CovidResNet()
model_image

class Classifier(nn.Sequential):
    """임베딩 된 feature를 이용해 classificaion
    """
    def __init__(self, model_image, **config):
        """
        Args:
            model_image : image emedding 모델
            config: 모델 설정 값
        """
        super(Classifier, self).__init__()

        self.model_image = model_image # image 임베딩 모델

        self.input_dim = model_image.num_ftrs # image feature 사이즈
        self.dropout = nn.Dropout(0.1) # dropout 적용

        self.hidden_dims = config['cls_hidden_dims'] # classifier hidden dimensions
        layer_size = len(self.hidden_dims) + 1 # hidden layer 개수
        dims = [self.input_dim] + [2] 

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)]) # classifer layers 

    def forward(self, v):
        # Drug/protein 임베딩
        v_i = self.model_image(v) # batch_size x hidden_dim 

        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor)-1):
                # If last layer,
                v_i = l(v_i)
            else:
                # If Not last layer, dropout과 ReLU 적용
                v_i = F.relu(self.dropout(l(v_i)))

        return v_i

model = Classifier(model_image, **config)
model


# hyperparameter
learning_rate = 0.0001
train_epoch   = 5
opt = torch.optim.Adam(model.parameters(), 
                       lr = learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()



import copy
from prettytable import PrettyTable
from time import time
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# train loop 
loss_history_train = []
loss_history_val = []

max_acc = 0
model = model.to(device)
model_best = copy.deepcopy(model) # best 모델 초기화 

# PrettyTable 
valid_metric_record = []
valid_metric_header = ['# epoch']
valid_metric_header.extend(['Accuracy', 'sensitivity', 'specificity', 'roc_score'])
table = PrettyTable(valid_metric_header)
float2str = lambda x:'%0.4f'%x # float 소숫점 4자리까지만 str로 바꾸기 

# 학습 진행
print('--- Go for Training ---')
# 학습 시작 시간 기록 
t_start = time() 


for epo in range(train_epoch):
    model.train()

    # mini batch training
    for v_i, label in train_loader:
        v_i = v_i.float().to(device) # input data gpu에 올리기 
        output = model(v_i) # forward-pass 
        loss = loss_fn(output, label.to(device)) # 미리 정의한 손실함수로 loss 계산, label ground truth 값 필요 
        loss_history_train.append(loss.item()) # 각 iteration마다 loss 기록 
        opt.zero_grad() # gradient 초기화 
        loss.backward() # back propagation
        opt.step() # parameter update 
    
    with torch.set_grad_enabled(False): # gradient tracking X -> valid 부분정의 
        y_pred = []
        y_score = []
        y_label = []
        model.eval()

        for i, (v_i, label) in enumerate(valid_loader):
            v_i = v_i.float().to(device) # validation input data gpu에 올리기 
            output = model(v_i) # forward pass 
            loss = loss_fn(output, label.to(device)) # 미리 정의한 손실함수로 loss 계산 
            loss_history_val.append(loss.item())

            pred = output.argmax(dim=1, keepdim=True) # 최대값 구하기 
            score = nn.Softmax(dim=1)(output)[:,1]

            # 예측값, 참ㄱ밧 cpu로 옮기고 numpy로 변형을 해줌 
            pred = pred.cpu().numpy()
            score = score.cpu().numpy()
            label = label.cpu().numpy()

            # 예측값, 참값 기록하기 
            y_label = y_label + label.flatten().tolist()
            y_pred = y_pred + pred.flatten().tolist()
            y_score = y_score + score.flatten().tolist()



    # metric 계산
    classification_metrics = classification_report(y_label, y_pred,
                                                   target_names = ['CT_NonCOVID', 'CT_COVID'],
                                                   output_dict= True)
    
    # sensitivity is the recall of the positive class
    sensitivity = classification_metrics['CT_COVID']['recall']

    # specificity is the recall of the negative class 
    specificity = classification_metrics['CT_NonCOVID']['recall']

    # accuracy
    accuracy = classification_metrics['accuracy']

    # confusion matrix
    conf_matrix = confusion_matrix(y_label, y_pred)

    # roc score
    roc_score = roc_auc_score(y_label, y_score)


    # 계산한 metric 합치기
    lst = ["epoch " + str(epo)] + list(map(float2str,[accuracy, sensitivity, specificity, roc_score]))

    # 각 epoch 마다 결과값 pretty table에 기록
    table.add_row(lst)
    valid_metric_record.append(lst)
    
    # mse 기준으로 best model 업데이트
    if accuracy > max_acc:
        # best model deepcopy 
        model_best = copy.deepcopy(model)
        # max MSE 업데이트 
        max_acc = accuracy

    # 각 epoch 마다 결과 출력 
    print('Validation at Epoch '+ str(epo + 1) + ' , Accuracy: ' + str(accuracy)[:7] + ' , sensitivity: '\
						 + str(sensitivity)[:7] + ', specificity: ' + str(f"{specificity}") +' , roc_score: '+str(roc_score)[:7])



# model test 

# Test dataloader 확인 
for i, (v_i, label) in enumerate(test_loader):
    print(v_i.shape)
    print(label.shape)
    break

# 테스트 진행

model = model_best

y_pred = []
y_label = []
y_score = []

model.eval()
for i, (v_i, label) in enumerate(test_loader):
    # input data gpu에 올리기 
    v_i = v_i.float().to(device)

    with torch.set_grad_enabled(False):
        # forward-pass
        output = model(v_i)

        # 미리 정의한 손실함수(MSE)로 손실(loss) 계산 
        loss = loss_fn(output, label.to(device))

        # 각 iteration 마다 loss 기록 
        loss_history_val.append(loss.item())

        pred = output.argmax(dim=1, keepdim=True)
        score = nn.Softmax(dim = 1)(output)[:,1]

        # 예측값, 참값 cpu로 옮기고 numpy 형으로 변환
        pred = pred.cpu().numpy()
        score = score.cpu().numpy()
        label = label.cpu().numpy()

    # 예측값, 참값 기록하기
    y_label = y_label + label.flatten().tolist()
    y_pred = y_pred + pred.flatten().tolist()
    y_score = y_score + score.flatten().tolist()

# metric 계산
classification_metrics = classification_report(y_label, y_pred,
                    target_names = ['CT_NonCOVID', 'CT_COVID'],
                    output_dict= True)
# sensitivity is the recall of the positive class
sensitivity = classification_metrics['CT_COVID']['recall']
# specificity is the recall of the negative class 
specificity = classification_metrics['CT_NonCOVID']['recall']
# accuracy
accuracy = classification_metrics['accuracy']
# confusion matrix
conf_matrix = confusion_matrix(y_label, y_pred)
# roc score
roc_score = roc_auc_score(y_label, y_score)

# 각 epoch 마다 결과 출력 
print('Validation at Epoch '+ str(epo + 1) + ' , Accuracy: ' + str(accuracy)[:7] + ' , sensitivity: '\
                            + str(sensitivity)[:7] + ' specificity: ' + str(f"{specificity}") +' , roc_score: '+str(roc_score)[:7])


# Roc curve 
# plot the roc curve    
fpr, tpr, _ = roc_curve(y_label, y_score)
plt.plot(fpr, tpr, label = "Area under ROC = {:.4f}".format(roc_score))
plt.legend(loc = 'best')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Confusion matrix plot 
import seaborn as sns

conf_matrix = conf_matrix
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, ax = ax, cmap = 'Blues'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['CoViD', 'NonCoViD']); ax.yaxis.set_ticklabels(['CoViD', 'NonCoViD'])