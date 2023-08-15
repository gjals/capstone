import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,f1_score
import pickle
from dataset import CustomDataset
from sklearn.model_selection import KFold
from fastai.losses import MSELossFlat
from fastai.optimizer import Adam
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tqdm import tqdm 

def train(lr=1e-3,wd=0,p=0,data=''):

    if data=='':
        #if dataset not selected excute program
        print('No specific data')
        exit()
    
    dir_path = '/content/drive/MyDrive/pytorch_study/Bunnit/train_torch'

    #load train data
    with open(dir_path + "/lunge271_round_1000.pickle.pkl","rb") as fr:
        train = pickle.load(fr)

    #load test data
    with open("test_torch/lunge20_round_1000.pickle.pkl","rb") as fr:
        test = pickle.load(fr)

    train_combined_label=[]  # (class label, count target)
    vaild_combined_label=[]  # (class label, count target)

    train_set=CustomDataset(np.array(train[0].permute(0,2,1)), np.array(train[2]))
    val_set=CustomDataset(np.array(test[0].permute(0,2,1)), np.array(test[2]))
    
    #make data loader for model trainig
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=10, shuffle=False)
    print(len(train_set))
    print(len(train_dataloader))
    print(len(val_set))
    print(len(val_dataloader))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Combine_model=InceptionTimePlus(c_in=8, c_out=1, seq_len=1000).to(device)
   
    #optimizer= Adam(Combine_model.parameters(), lr=lr) 
    optimizer=torch.optim.Adam(Combine_model.parameters(),lr=lr,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    count_criterion=MSELossFlat()
    #traning step
    epochs= 60
    best_train_acc= 0.0
    best_test_acc= 0.0

    #start training
    for epoch in tqdm(range(epochs)):

        #local variable for result check
        count_loss_sum=0
        count_true_labels=[]
        count_pred_labels=[]

        #training step
        Combine_model.train()
        
        for e_num,(x,y) in enumerate(train_dataloader):
            batch_count=y
            x,batch_count=x.to(device),batch_count.float().to(device) #data to device
            
            Combine_model.zero_grad()
            
            pred_count=Combine_model(x) 
            count_true_labels.extend(batch_count.cpu().numpy())
            count_pred_labels.extend(np.round(pred_count.detach().cpu().numpy()))
            
            #calculate loss
            count_loss=count_criterion(pred_count, batch_count)
            #loss result append
            count_loss_sum+=count_loss.detach().item()
        
            #combine loss for gradient calculate and update model parameter
            count_loss.backward()
            optimizer.step()

        #calculate training step result  
        mse= mean_squared_error(count_true_labels,count_pred_labels)    
        train_acc=accuracy_score(list(map(int, count_true_labels)), list(map(int, count_pred_labels)))
        print(f'train \t\t count loss mean {round(count_loss_sum/e_num,3)}  MSE :{round(mse,3)}, count_acc: {train_acc}\n')
        
        #local variable for result check
        count_loss_sum=0
        count_true_labels=[]
        count_pred_labels=[]
        
        #validation dataset verify
        Combine_model.eval()
        for e_num,(x,y) in enumerate(val_dataloader):
            batch_count=y
            x,batch_count=x.to(device),batch_count.float().to(device)
            pred_count=Combine_model(x)
            pred_count= pred_count.reshape(-1)
           
            count_true_labels.extend(batch_count.cpu().numpy())
            count_pred_labels.extend(np.round(pred_count.detach().cpu().numpy()))
            
            #calculate loss, not for train just for check result 
            count_loss=count_criterion(pred_count, batch_count)
            count_loss_sum+=count_loss.detach().item()    

       #calculate validation step result    
        mse= mean_squared_error(count_true_labels,count_pred_labels)
        test_acc=accuracy_score(list(map(int, count_true_labels)), list(map(int, count_pred_labels)))
        
        print(f'validataion \t count loss mean {round(count_loss_sum/e_num,3)}  MSE :{round(mse,3)}, count_acc: {test_acc}')    
        print(count_true_labels)
        print(count_pred_labels)
        
        scheduler.step()

        if best_test_acc < test_acc or (best_test_acc==test_acc and train_acc>best_train_acc) :
            print(f'update train acc {best_train_acc}->{train_acc}')
            print(f'update test acc {best_test_acc}->{test_acc}')
            best_train_acc= train_acc
            best_test_acc= test_acc
            model_name= 'burpee_round3_model'
            torch.save(Combine_model.state_dict(), 'model_save/'+ model_name +'.pt')
            
    #trainig end 

if __name__ == "__main__":
    train(lr=0.005)