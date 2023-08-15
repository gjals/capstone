import torch
import torch.nn as nn
import numpy as np
from util import *
from scipy.signal import lfilter
from InceptionTimePlus_1 import *
import coremltools as ct

def find_class(file):
    if 'squat' in file:
        return 0
    elif 'lunge' in file:
        return 1
    elif 'situp' in file:
        return 2
    elif 'burpee' in file:
        return 3

def decode(data):  # 8가지 값 사용
    data_dict = {'accX': data.WatchAccX, 'accY': data.WatchAccY, 'accZ': data.WatchAccZ, 'angX': data.WatchGyroX, 'angY': data.WatchGyroY, 'angZ': data.WatchGyroZ}
    new_data = pd.DataFrame(data=data_dict)
    new_data['acc_scala'] = (new_data['accY'] ** 2 + new_data['accX'] ** 2 + new_data['accZ'] ** 2).apply(np.sqrt)[:].rolling(3, min_periods=1, center=True).mean()
    new_data['ang_scala'] = (new_data['angY'] ** 2 + new_data['angX'] ** 2 + new_data['angZ'] ** 2).apply(np.sqrt)[:].rolling(3, min_periods=1, center=True).mean()
    
    return new_data

def sampling(data, sampling_num):
    sampled_data = []
    data_col=data.columns
    data=data.to_numpy()
    for row in range(0, len(data), sampling_num):  
        if row + sampling_num > len(data): 
            extra_row = row + sampling_num - len(data) 
            sampled_data.append(np.mean(data[row:row+extra_row],axis=0))
        else:
            sampled_data.append(np.mean(data[row:row+sampling_num,:],axis=0))  
    return pd.DataFrame(sampled_data,columns=data_col)

def save_torch_raw_data(dir_path, file_name):
    
    max_len = 1000
    columns_list= ['WatchAccX', 'WatchAccY', 'WatchAccZ', 'WatchGyroX', 'WatchGyroY', 'WatchGyroZ']
    column_list= ['accX','accY','accZ','angX','angY','angZ','acc_scala','ang_scala']

    data = pd.read_csv(dir_path + file_name, index_col=False, usecols= column_list)

    if(np.isnan(file['WatchAccX'][0])): #빈 파일인지 체크
        return ''

    file= file[~(np.isnan(file['WatchAccX']))] #파일의 빈 부분 없애기
    data = decode(file) 
    
    filter_num= int(np.ceil(len(data[data.columns[0]])/max_len))
   
    # smoothing
    data = data.rolling(filter_num).mean() #filter_num개의 요소마다 평균내줌 [1,2,3]->[nan,nan,2]
   
    # sampling
    data = sampling(data, sampling_num=filter_num) 

    n = 5  
    b = [1.0 / n] * n
    a = 1
    for col in data.columns:
        data[col]= lfilter(b, a, data[col])

    data_n6= pd.DataFrame()
    n = 6 
    b = [1.0 / n] * n
    a = 1
    for col in data.columns:
        data_n6[col]= lfilter(b, a, data[col])
    
    # normalization
    data = (data - data.mean())/data.std()
    data_n6 = (data_n6 - data_n6.mean())/data_n6.std()
      
    if data.isnull().values.any():#null이 있으면 0으로 채우기
        data = data.fillna(0)   
        data_n6 = data_n6.fillna(0)

    if len(data) < max_len:  
        pad = pd.DataFrame(np.zeros((max_len - len(data), 8)),columns=data.columns)
        cat_df = pd.concat([data, pad], axis=0).reset_index() 
        cat_df = cat_df.drop(['index'], axis=1) 

        cat_df_n6 = pd.concat([data_n6, pad], axis=0).reset_index() 
        cat_df_n6 = cat_df_n6.drop(['index'], axis=1) 

    x_data = torch.tensor(cat_df.values).float()
    x_data_n6 = torch.tensor(cat_df_n6.values).float()
    y_label = find_class(file_name)
    y_count = int(file_name.split('_')[2])

    return x_data, x_data_n6, y_label, y_count


def execute_class_model(dir, model_filename, data):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data= data.permute(1,0)[None, :, :] #[1000,2]->[1, 2, 1000]

    model_dict = torch.load(dir+model_filename, map_location=device)
    inception_model=InceptionTimePlus(c_in= 8, c_out=4, seq_len= 1000).to(device)
    inception_model.load_state_dict(model_dict)
    inception_model.eval()
    scripted_model= torch.jit.trace(inception_model, data)

    pred_value = inception_model(data) #tensor([[ 0.0188, -0.5396, -0.0150, -0.0120]], grad_fn=<AddmmBackward0>)처럼 출력
    scripted_pred_value = scripted_model(data)
    print(pred_value, scripted_pred_value)
    
    #model_from_torch = ct.convert(scripted_model, inputs=[ct.TensorType(name="input", shape=data.shape)])
    #model_from_torch.save("mlmodel/all_model.mlmodel")
   
    return pred_value.reshape(-1).softmax(dim=0).argmax(dim=0).item()

def execute_count_model(dir, model_filename, data):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data= data.permute(1,0)[None, :, :] #[1000,2]->[1, 2, 1000]

    model_dict = torch.load(dir+model_filename, map_location=device)
    inception_model=InceptionTimePlus(c_in= 8, c_out=1, seq_len= 1000).to(device)
    inception_model.load_state_dict(model_dict)
    inception_model.eval()
    scripted_model= torch.jit.trace(inception_model, data)

    pred_value = inception_model(data) #tensor([[1.7888]], grad_fn=<AddmmBackward0>) 출력
    scripted_pred_value = scripted_model(data)
    print(pred_value, scripted_pred_value)

    #model_from_torch = ct.convert(scripted_model, inputs=[ct.TensorType(name="input", shape=data.shape)])
    #model_from_torch.save("mlmodel/lunge_round_model.mlmodel")
    
    return pred_value.reshape(-1).item()

def fun(file):
    if 'csv' in file:
        return True  

def getfiles(dir_path):
    files = os.listdir(dir_path) #주소의 파일을 리스트로 반환
    return list(filter(fun, files)) #txt 파일 리스트 반환

if __name__ == "__main__":

    data_path= 'C:/Users/82103/Desktop/exer_data/'#pytorch_study/raw_train/'
    model_path= 'C:/Users/82103/Desktop/pytorch_study/Bunnit-WorkoutClassification_and_Counting/models/'
    csv_filename= '1_lunge_16_1.csv'
    
    class_model_filename= 'all_model.pt'
    squat_model_filename= 'squat_round_model.pt'
    lunge_model_filename= 'lunge_round2_model.pt' 
    situp_model_filename= 'situp_round_model.pt' 
    burpee_model_filename= 'burpee_round2_model.pt' 

    print("Preprocessing raw sensordata...")
    x_data, x_data_n6, y_label, y_count= save_torch_raw_data(data_path, csv_filename)
    print("Successfully save preprocessed torch data!")
    
    pred_class= execute_class_model(model_path, class_model_filename, x_data)
    
    if pred_class==0:
        pred_count= execute_count_model(model_path, squat_model_filename, x_data_n6)
    elif pred_class==1:
        pred_count= execute_count_model(model_path, lunge_model_filename, x_data)
    elif pred_class==2:
        pred_count= execute_count_model(model_path, situp_model_filename, x_data_n6)
    elif pred_class==3:
        pred_count= execute_count_model(model_path, burpee_model_filename, x_data)

    print(pred_class, pred_count)


    
