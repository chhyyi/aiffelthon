#!/usr/bin/env python
# coding: utf-8

# # GRU 써보기    
# 이 노트북은 [Sensory_LSTM 노트북](https://github.com/chhyyi/aiffel/blob/main/aiffelthon/Sensory_LSTM.ipynb) 의 복제입니다.
# 
# ## 작업 로그
# 
# ### 2022-11-12
# - 지난번에 하려고 했던, wandb를 이용해서 sweep을 해봅니다. wandb에 의한 sweep은 python module로 저장하고 임포트 하는 것이 더 적당해 보입니다. 이 부분은 천천히 생각합니다. 아무튼 그 경우, '\*.py' 파일을 내보내고 여기에서 wandb sweep 을 실행하도록 합니다.  
# 
# ### 2022-11-10
# - 원본의 LSTM을 GRU로 바꾸고, wandb를 써보려고 했는데 그냥 한 번 했습니다. 
# - df_merged를 csv로 저장하고, 시퀀스 인덱스 생성 이전의 부분은 모두 삭제합니다. csv파일을 읽는 부분부터 씁니다.

# In[1]:

def train():
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    import matplotlib.pyplot as plt
    import numpy as np
    import wandb
    from wandb.keras import WandbCallback
    
    def seq_acc(y_true, y_pred):
        y_bin=np.zeros_like(y_pred)
        for i, dd in enumerate(y_bin):
            for j in range(len(dd)):
                pred=y_pred[i][j]
                if pred>=0.5:
                    y_bin[i][j]=1
                else:
                    y_bin[i][j]=0

        predict_true = (y_true == y_bin)
        # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
        try:
            score = np.average(np.average(predict_true))
        except ValueError:
            score = mean_squared_error(y_true, y_bin)
        return score

    def my_seq_acc(y_true, y_pred):
        score = tf.py_function(func=seq_acc, inp=[y_true, y_pred], Tout=tf.float32,  name='custom_seq_acc') # tf 2.x
        #score = tf.py_func( lambda y_true, y_pred : mse_AIFrenz(y_true, y_pred) , [y_true, y_pred], 'float32', stateful = False, name = 'custom_mse' ) # tf 1.x
        return score


    # In[2]:

    class MySeqAccCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epochs, logs=None):
            y_pred=self.model.predict(X_test)
            print('sequence accuracy is {}'.format(seq_acc(y_test, y_pred)))



    wandb.login()


    # In[ ]:


    run = wandb.init(project = 'redzone_gru',
                     entity = 'chhyyi',
                     config = {
                         'seq_field':72,
                         'stride_inside_seq':9,
                         'stride_between_seqs':2,
                         'learning_rate':0.01,
                         'split_train_ratio':0.8,
                         'epochs':20,
                         'batch_size':64,
                         'unit_gru0':64,
                         'unit_gru1':64,
                         'callbacks':[MySeqAccCallback()],
                     })


    # In[75]:


    locations=['거문도', '울산', '거제도', '통영', '추자도']


    # In[76]:


    # load normalized data

    df_merged=pd.read_csv("sensory_preprocessed_df.csv")
    if df_merged.columns[0]=='Unnamed: 0':
        df_merged = df_merged.iloc[:, 1:]

    print('loaded dataset. Generating sequences')
    seq_length=wandb.config.seq_field//wandb.config.stride_inside_seq
    len_ds=len(df_merged)

    seqs_idx=[]

    start_idx=0
    while start_idx<=len_ds-wandb.config.seq_field:
        seqs_idx.append(list(range(start_idx, start_idx + wandb.config.seq_field, wandb.config.stride_inside_seq
    )))
        start_idx+=wandb.config.stride_between_seqs


    seqs_idx[100],len(seqs_idx[100])

    df_merged.reset_index(inplace=True, drop=True)
    print('no missing values:', df_merged.isna().any().any())


    #train_cols=['풍속(m/s)', '풍향(deg)', '기온(°C)', '수온(°C)', '강수량(mm)', '적조발생']
    ds_train_cols=df_merged
    ds_train_cols.reset_index(inplace=True, drop=True)
    print('train dataset columns:',ds_train_cols.columns)

    seq_dataset=np.zeros([len(seqs_idx), len(seqs_idx[0]), len(ds_train_cols.columns)])

    for i, seq in enumerate(seqs_idx):
        for j, row_number in enumerate(seq):
            seq_dataset[i, j]=ds_train_cols.loc[row_number].to_numpy()

    def not_bin_in_occurence(x):
        if x==1 or x==0:
            return x
        else:
            print('exceptional value(not 0 or 1) found. replaced by near one.')
            if x>=0.5:
                return 1
            else:
                return 0
            
    ds_train_cols['적조발생']=ds_train_cols['적조발생'].apply(not_bin_in_occurence)


    split_index=int(len(seq_dataset)*wandb.config.split_train_ratio)
    print(split_index, len(seq_dataset))

    train_xy=seq_dataset[:split_index]
    np.random.shuffle(train_xy)
    X_train=train_xy[:,:,0:-1]
    y_train=train_xy[:,:,-1]

    test_xy=seq_dataset[split_index:]
    np.random.shuffle(test_xy)
    X_test=test_xy[:,:,0:-1]
    y_test=test_xy[:,:,-1]

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape,'\n\n')

    model = keras.Sequential([
        keras.Input(shape=(seq_length, 25)),
        keras.layers.GRU(wandb.config.unit_gru0, return_sequences=True),
        keras.layers.GRU(wandb.config.unit_gru1),
        keras.layers.Dense(8, activation="sigmoid"),
    ]
    )


    # In[95]:
    optimizer=keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)

    model.compile(optimizer=optimizer, loss="binary_crossentropy")

    # In[97]:


    # In[98]:


    model.fit(X_train, y_train,
            batch_size=wandb.config.batch_size,
            epochs=wandb.config.epochs, 
            validation_data=(X_test, y_test),
            callbacks=wandb.config.callbacks,
             )

    wandb.finish()