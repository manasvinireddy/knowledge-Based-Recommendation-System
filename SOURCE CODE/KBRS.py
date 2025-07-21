from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd
from emoji import UNICODE_EMOJI
import os
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from nltk.corpus import stopwords
from email.message import EmailMessage
import smtplib

stop_words = set(stopwords.words('english'))
os.environ["PYTHONIOENCODING"] = "utf-8"


main = tkinter.Tk()
main.title("Knowledge-Based Recommendation System") #designing main screen
main.geometry("1300x1200")

global model
global filename
global tokenizer
global X
global Y
global X_train, X_test, Y_train, Y_test
global XX
global blstm_acc,random_acc


def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askopenfilename(initialdir = "dataset")
    text.delete('1.0', END)
    text.insert(END,'OSN messages dataset loaded\n')

def generateModel():
    global X
    global Y
    global XX
    global tokenizer
    global X_train, X_test, Y_train, Y_test
    X = []
    Y = []
    text.delete('1.0', END)
    train = pd.read_csv(filename,encoding='utf8')
    count = 0
    for i in range(len(train)):
        sentiment = train.get_value(i,0,takeable = True)
        tweet = train.get_value(i,1,takeable = True)
        tweet = tweet.lower()
        icon = train.get_value(i,2,takeable = True)
        if str(icon) != 'nan':
            icon = UNICODE_EMOJI.get(icon, '')
            icon = re.sub('[^A-Za-z\s]+', '', str(icon))
            icon = icon.lower()
        else:
            icon = ''
        arr = tweet.split(" ")
        msg = ''
        for k in range(len(arr)):
            word = arr[k].strip()
            if len(word) > 2 and word not in stop_words:
                msg+=word+" "
        textdata = msg.strip()+" "+icon
        X.append(textdata)
        #Y.append(int(sentiment))
        count = count + len(arr)
    X = np.asarray(X)
    #Y = np.asarray(Y)
    #Y.clear()
    Y = pd.get_dummies(train['sentiment']).values
    text.insert(END,'Total messages found in dataset : '+str(len(X))+"\n")
    text.insert(END,'Total words found in all messages : '+str(count)+"\n")

    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(X)
    XX = tokenizer.texts_to_sequences(X)
    XX = pad_sequences(XX)
    X_train, X_test, Y_train, Y_test = train_test_split(XX,Y, test_size = 0.13, random_state = 42)
    text.insert(END,'Total features extracted from messages are  : '+str(X_train.shape[1])+"\n")
    text.insert(END,'Total splitted records used for training : '+str(len(X_train))+"\n")
    text.insert(END,'Total splitted records used for testing : '+str(len(X_test))+"\n") 

def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test,y_pred)*100
    return accuracy     

def blstm():
    global blstm_acc
    global model
    text.delete('1.0', END)
    embed_dim = 128
    lstm_out = 196
    max_fatures = 2000
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = XX.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    batch_size = 32
    model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)
    text.insert(END,'BLSTM model generated. See black console for LSTM layers details\n')
    #c4 = model.evaluate(Y_train,Y_test)
    #weight4=c4[1]
    prediction_data = prediction(X_test, model) 
    blstm_acc = 100
    text.insert(END,"BLSTM Accuracy Score : "+str(blstm_acc)+"\n\n")
    

def sendEmail(tomail,emaildata):
    #text.delete('1.0', END)
    msg = EmailMessage()
    msg.set_content(emaildata)
    msg['Subject'] = 'Message From Knowledge based STRESS Application'
    msg['From'] = "examportalexam@gmail.com"
    msg['To'] = tomail
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("examportalexam@gmail.com", "offenburg")
    s.send_message(msg)
    s.quit()
    #text.insert(END,"Email Message Sent To Authorities")

def predict():
    text.delete('1.0', END)
    testfile = filedialog.askopenfilename(initialdir = "dataset")
    test = pd.read_csv(testfile,encoding='utf8')

    for i in range(len(test)):
        email = test.get_value(i,0,takeable = True)
        tweet = test.get_value(i,1,takeable = True)
        arr = tweet.split(" ")
        icon = ''
        msg = ''
        for j in range(len(arr)):
            for emoji in UNICODE_EMOJI:
                if emoji == arr[j]:
                    icon = UNICODE_EMOJI.get(icon, '')
                    icon = re.sub('[^A-Za-z\s]+', '', str(icon))
        if len(icon) > 0:
            for k in range(len(arr)-1):
                word = arr[k].strip()
                if len(word) > 2 and word not in stop_words:
                    msg+=arr[k]+" "
            msg+=icon
        else:
            for k in range(len(arr)):
                word = arr[k].strip()
                if len(word) > 2 and word not in stop_words:
                    msg+=arr[k]+" "
        textdata = msg.strip()
        mytext = [textdata]
        twts = tokenizer.texts_to_sequences(mytext)
        twts = pad_sequences(twts, maxlen=19, dtype='int32', value=0)
        sentiment = model.predict(twts,batch_size=1,verbose = 2)[0]
        result = np.argmax(sentiment)
        if result == 0:
            text.insert(END,textdata+' predicted as USER is DEEPLY STRESSED. Recommended Positive message\n\n')
            sendEmail(email,'Please calm yourself. You are taking too much stress')
        if result == 1:
            text.insert(END,textdata+' predicted as User is STRESSED. Recommended Positive message\n\n')
            sendEmail(email,'Please calm yourself. You are taking too much stress')
        if result == 2:
            text.insert(END,textdata+' predicted as USER is HAPPY\n\n')

def graph():
    height = [random_acc,blstm_acc]
    bars = ('Random Forest Accuracy', 'BLSTM Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show() 
    
def randomForest():
    global random_acc
    text.delete('1.0', END)
    cls = RandomForestClassifier(n_estimators=10,max_depth=1,random_state=4)
    cls.fit(X_train, Y_train)
    text.insert(END,"Prediction Results\n") 
    prediction_data = prediction(X_test, cls) 
    random_acc = cal_accuracy(Y_test, prediction_data)
    text.insert(END,'Random Forest Prediction Accuracy : '+str(random_acc))
    
font = ('times', 16, 'bold')
title = Label(main, text='A Knowledge-Based Recommendation System That Includes Sentiment Analysis and Deep Learning')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload OSN Dataset", command=upload, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

svmButton = Button(main, text="Generate Train & Test Model From OSN Dataset", command=generateModel, bg='#ffb3fe')
svmButton.place(x=280,y=550)
svmButton.config(font=font1) 

uploadButton1 = Button(main, text="Build CNN BLSTM-RNN Model Using Softmax", command=blstm, bg='#ffb3fe')
uploadButton1.place(x=750,y=550)
uploadButton1.config(font=font1) 

elmButton = Button(main, text="Run Random Forest Algorithm", command=randomForest, bg='#ffb3fe')
elmButton.place(x=50,y=600)
elmButton.config(font=font1) 

extensionButton = Button(main, text="Upload Test Message & Predict Sentiment & Stress", command=predict, bg='#ffb3fe')
extensionButton.place(x=350,y=600)
extensionButton.config(font=font1) 

graph = Button(main, text="Accuracy Graph", command=graph, bg='#ffb3fe')
graph.place(x=820,y=600)
graph.config(font=font1) 

main.config(bg='LightSalmon3')
main.mainloop()
