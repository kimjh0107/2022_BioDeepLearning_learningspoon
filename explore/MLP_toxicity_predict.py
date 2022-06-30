"""
Dataset Description: Acute toxicity LD50 measures the most conservative dose that can lead to lethal adverse effects. The higher the dose, the more lethal of a drug. This dataset is kindly provided by the authors of [1].

Task Description: Regression. Given a drug SMILES string, predict its acute toxicity.

Dataset Statistics: 7,385 drugs.
"""
from tdc.single_pred import Tox 
# LD50 data using TDC api 
data = Tox(name = 'LD50_Zhu')
split = data.get_split() # Tox 구현 함수 
split.keys() # data check 
split['train']

# %% 
# Data pre-processing 
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np 

# SMILES data -> morgan fingerprint 로 데이터 변환 
def smiles2morgan(s, radius = 2, nBits = 1024):

    """
    Args: 
        s: SMILES of drug 
        radius: ECFP radius 
        bBits: size of binary representation 
    Return:
        morgan fingerprint 
    """
    try: 
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print('rdkit not found this similes for morgan:' + s + 'convert to all 0 features')
        features = np.zeros((nBits,))
    return features 

# 전처리 함수 적용하기 
for mode in ['train', 'valid', 'test']:
    split[mode]['embedding'] = split[mode]['Drug'].apply(smiles2morgan)

# 변환 데이터 확인 
split['test']['embedding']

# %%
# Torch dataset & loaders 
import torch 
from torch import embedding, nn 
import torch.nn.functional as F 
from torch.utils import data 

class data_process_loader(data.Dataset):
    def __init__(self, df):
        self.df = df 
    
    def __len__(self):
        return self.df.shape[0] # 샘플 개수 출력을 위한 method 
    
    def __getitem__(self, index):
        v_d = self.df.iloc[index]['embedding'] # input 
        y = self.df.iloc[index]['Y'] # label 
        return v_d, y

train_dataset = data_process_loader(split["train"])
valid_dataset = data_process_loader(split["valid"])
test_dataset = data_process_loader(split["test"])

# hyperparameter 
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 1,
          'drop_last': False}

training_generator = data.DataLoader(train_dataset, **params)
valid_generator = data.DataLoader(valid_dataset, **params)
test_generator = data.DataLoader(test_dataset, **params)       

# 데이터 확인 
for v_d,y in training_generator:
    print(v_d)
    print(v_d.shape)
    print(y)
    print(y.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Build up model 
# MLP
class MLP(nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_dims_lst):
        super(MLP, self).__init__()

        # feature extractor layer size 
        layer_size = len(hidden_dims_lst) + 1 
        # 각 층의 차원 크기를 담는 리스트 
        dims = [input_dim] + hidden_dims_lst + [output_dim]
        # input, hidden layer, output layer 차원대로 linear layer 쌓기 
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

    def forward(self, v):
        # input data 'v' forward pass 
        v = v.float().to(device)
        for l in self.predictor:
            v = F.relu(l(v))
        return v 


class Classifier(nn.Sequential):
	def __init__(self, model_drug, hidden_dim_drug, cls_hidden_dims):
		'''Classifier
			Args:
				model_drug : 앞서 생성한 Feature extractor 
				hidden_dim_drug (int): Classifier 입력층 차원
				cls_hidden_dims (list): Classifier hidden 차원
		'''
		super(Classifier, self).__init__()
  
		# feature extractor
		self.model_drug = model_drug

		# dropout
		self.dropout = nn.Dropout(0.1)
  
		# classifier 입력 차원
		self.input_dim_drug = hidden_dim_drug

		# classifier hidden 차원
		self.hidden_dims = cls_hidden_dims

		# classifier layer size 
		layer_size = len(self.hidden_dims) + 1

		# 각 층의 차원 크기를 담은 리스트 
		dims = [self.input_dim_drug] + self.hidden_dims + [1]
		
		# 입력층, hidden 층, 출력층 차원대로 linear layer 쌓기
		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

	def forward(self, v_D):
		# feature extractor로 입력 데이터 encoding
		v_f = self.model_drug(v_D)
		
		# forword-pass with classify 
		for i, l in enumerate(self.predictor):
			if i==(len(self.predictor)-1):
				v_f = l(v_f)
			else:
				v_f = F.relu(self.dropout(l(v_f)))
		return v_f 

# 모델 hyperparameter
input_dim_drug = 1024
hidden_dim_drug = 256
cls_hidden_dims = [1024, 1024, 512]
mlp_hidden_dims_drug = [1024, 256, 64]

model_drug = MLP(1024, hidden_dim_drug, mlp_hidden_dims_drug)
model = Classifier(model_drug, hidden_dim_drug, cls_hidden_dims)
model


# %%
# model run 
# 학습 진행에 필요한 hyperparameter 
learning_rate = 0.0001
decay         = 0.00001
train_epoch   = 15

# optimizer 
opt = torch.optim.Adam(model.parameters(), 
                       lr = learning_rate, 
                       weight_decay = decay)
loss_fn = torch.nn.MSELoss()


import copy
from prettytable import PrettyTable
from time import time
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from lifelines.utils import concordance_index

loss_history = []
max_MSE = 1000000

model = model.to(device)

# best model 초기화 
model_max = copy.deepcopy(model)

# 결과 정리를 위한 *prettytable!!
valid_metric_record = []
valid_metric_header = ['#epoch']
valid_metric_header.extend(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
table = PrettyTable(valid_metric_header)

float2str = lambda x:'%0.4f'%x # float 소숫점 4자리까지만 str 로 변경 

# 학습 진행
print('--- Go for Training ---')
# 학습 시작 시간 기록 
t_start = time() 

for epo in range(train_epoch):
    # model training
    model.train()
    # mini-batch training 
    for i, (v_d, label) in enumerate(training_generator):
        v_d = v_d.float().to(device) # input data gpu에 올리기 
        # forward pass 
        score = model(v_d)
        n = torch.squeeze(score, 1)

        # loss 계산 
        loss = loss_fn(n.float(), label.float().to(device))
        # iteration마다 loss기록 
        loss_history.append(loss.item())
        # gradient 초기화 
        opt.zero_grad()
        # back propagation
        loss.backward()
        # parameters update 
        opt.step()
    # gradient tracking X
    with torch.set_grad_enabled(False):
        
        y_pred = []
        y_label = []
        # model validation
        model.eval()

        for i, (v_d, label) in enumerate(valid_generator):
            # validation 입력 데이터 gpu에 올리기
            v_d = v_d.float().to(device)

            # forward-pass
            score = model(v_d)

            # 예측값, 참값 cpu로 옮기고 numpy 형으로 변환
            logits = torch.squeeze(score).cpu().numpy()
            label_ids = label.cpu().numpy()

            # 예측값, 참값 기록하기
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()   

    # metric 계산
    mse = mean_squared_error(y_label, y_pred) # MSE 
    r2 = pearsonr(y_label, y_pred)[0] # Pearson correlation coefficient
    p_val = pearsonr(y_label, y_pred)[1] # Pearson correlation p-value
    CI =  concordance_index(y_label, y_pred) # CI 

    # 계산한 metric 합치기
    lst = ["epoch " + str(epo)] + list(map(float2str,[mse, r2, p_val, CI]))

    # 각 epoch 마다 결과값 pretty table에 기록
    table.add_row(lst)
    valid_metric_record.append(lst)

    # mse 기준으로 best model 업데이트
    if mse < max_MSE:
        # best model deepcopy 
        model_max = copy.deepcopy(model)
        # max MSE 업데이트 
        max_MSE = mse

    # 각 epoch 마다 결과 출력 
    print('Validation at Epoch '+ str(epo + 1) + ' , MSE: ' + str(mse)[:7] + ' , Pearson Correlation: '\
						 + str(r2)[:7] + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI)[:7])


# %%
# 테스트 진행

y_pred = []
y_label = []

model.eval()
for i, (v_d, label) in enumerate(test_generator):
    # input data gpu에 올리기 
    v_d = v_d.float().to(device)

    # forward-pass
    score = model(v_d)

    # 예측값 gradient graph detach -> cpu로 옮기기 -> numpy 형으로 변환 
    logits = torch.squeeze(score).detach().cpu().numpy()

    # 참값 cpu로 옮기고 numpy 형으로 변환 
    label_ids = label.cpu().numpy()

    # 예측값, 참값 기록
    y_label = y_label + label_ids.flatten().tolist()
    y_pred = y_pred + logits.flatten().tolist()

# metric 계산
mse = mean_squared_error(y_label, y_pred)
r2 = pearsonr(y_label, y_pred)[0]
p_val = pearsonr(y_label, y_pred)[1]
CI =  concordance_index(y_label, y_pred)

print('TestSet Performence Metric '+  ' , MSE: ' + str(mse)[:7] + ' , Pearson Correlation: '\
        + str(r2)[:7] + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI)[:7])


# 최종 테스트 결과 시각화 
import matplotlib.pyplot as plt

# 참값 ~ 예측값 scatter plot 
plt.figure(figsize=(10,10))
plt.scatter(y_label, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_pred), max(y_label))
p2 = min(min(y_pred), min(y_label))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.title('Chemical Toxicity Prediction')
plt.show()