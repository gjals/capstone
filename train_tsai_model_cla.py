import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,f1_score
import pickle
from dataset import CustomDataset
from sklearn.model_selection import KFold
from fastai.losses import CrossEntropyLossFlat
from fastai.optimizer import Adam
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tqdm import tqdm 

def train(lr=1e-3,wd=0,p=0):
    if data=='':
        #if dataset not selected excute program
        print('No specific data')
        exit()
    

    dir_path = '/content/drive/MyDrive/pytorch_study/Bunnit/train_torch'
    #load train data
    with open(dir_path + "/all_1000.pickle.pkl","rb") as fr:
        train = pickle.load(fr)

    #load test data
    with open("test_torch/all102_1000.pickle.pkl","rb") as fr:
        test = pickle.load(fr)

    train_combined_label=[]  # (class label, class target)
    vaild_combined_label=[]  # (class label, class target)

    train_set=CustomDataset(np.array(train[0].permute(0,2,1)), np.array(train[1]))
    val_set=CustomDataset(np.array(test[0].permute(0,2,1)), np.array(test[1]))
    
    #make data loader for model trainig
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=10, shuffle=False)
    print(len(train_set))
    print(len(train_dataloader))
    print(len(val_set))
    print(len(val_dataloader))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #model,optimizer,scheduler and loss functions Declaration
    Combine_model=InceptionTimePlus(c_in=8, c_out=4, seq_len= 1000).to(device)
    optimizer=torch.optim.Adam(Combine_model.parameters(),lr=lr,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    class_criterion=CrossEntropyLossFlat()
    #traning step
    epochs= 30
    best_train_acc= 0.0
    best_test_acc= 0.0

    for epoch in tqdm(range(epochs)):

        #local variable for result check
        class_loss_sum=0
        class_true_labels=[]
        class_pred_labels=[]

        #training step
        Combine_model.train()
        
        for e_num,(x,y) in enumerate(train_dataloader):
            batch_class=y
            x,batch_class=x.to(device),batch_class.long().to(device) #data to device
            
            Combine_model.zero_grad()
            pred_class=Combine_model(x) 

            class_pred_labels.extend(pred_class.softmax(dim=-1).argmax(dim=-1).cpu().numpy())
            class_true_labels.extend(batch_class.cpu().numpy())
            #class_pred_labels.extend(np.round(pred_class.detach().cpu().numpy()))
            
            #calculate loss
            class_loss=class_criterion(pred_class, batch_class)
            #loss result append
            class_loss_sum+=class_loss.detach().item()
        
            #combine loss for gradient calculate and update model parameter
            class_loss.backward()
            #optimizer.zero_grad()
            optimizer.step()

        #calculate training step result  
        mse= mean_squared_error(class_true_labels,class_pred_labels)    
        train_acc=accuracy_score(list(map(int, class_true_labels)), list(map(int, class_pred_labels)))
        t_f1=f1_score(list(class_true_labels), class_pred_labels, average='macro')
        print(f'train \t\t class loss mean {round(class_loss_sum/e_num,3)}  MSE :{round(mse,3)}, class_acc: {train_acc}')
        print(f't_f1:{t_f1}')

        #local variable for result check
        class_loss_sum=0
        class_true_labels=[]
        class_pred_labels=[]
        
        #validation dataset verify
        Combine_model.eval()
        for e_num,(x,y) in enumerate(val_dataloader):

            batch_class=y
            x,batch_class=x.to(device),batch_class.long().to(device)
            pred_class=Combine_model(x)
        
            class_pred_labels.extend(pred_class.softmax(dim=-1).argmax(dim=-1).cpu().numpy())
            class_true_labels.extend(batch_class.cpu().numpy())
           
            #calculate loss, not for train just for check result 
            class_loss=class_criterion(pred_class, batch_class)
            class_loss_sum+=class_loss.detach().item()      

        #calculate validation step result    
        mse= mean_squared_error(class_true_labels,class_pred_labels)
        test_acc=accuracy_score(list(map(int, class_true_labels)), list(map(int, class_pred_labels)))
        v_f1=f1_score(list(class_true_labels), class_pred_labels, average='macro')

        print(f'validataion \t class loss mean {round(class_loss_sum/e_num,3)}  MSE :{round(mse,3)}, class_acc: {test_acc}',end='\n')
        print(f'v_f1:{v_f1}')
        print(class_true_labels)
        print(class_pred_labels)


        if best_test_acc < test_acc or (best_test_acc==test_acc and train_acc>best_train_acc) :
            print(f'update train acc {best_train_acc}->{train_acc}')
            print(f'update test acc {best_test_acc}->{test_acc}')
            best_train_acc= train_acc
            best_test_acc= test_acc
            model_name= 'all_model'
            torch.save(Combine_model.state_dict(), 'model_save/'+ model_name +'.pt')
         
        #scheduler.step()
    #trainig end 

if __name__ == "__main__":

    train(lr=0.001)