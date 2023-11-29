import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import gensim.downloader as api
import warnings
warnings.filterwarnings('ignore')
import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import argparse



parser = argparse.ArgumentParser()


parser.add_argument('-d', '--data_dir', type=str, default='filtered_corpusid_input.csv')
parser.add_argument('-T', '--text_feature_col', type=str, default='title')
parser.add_argument('-w', '--w2v_word_count', type=int, default=100)
parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
parser.add_argument('-e', '--epoch', type=int, default=10)
parser.add_argument('-b', '--batch_size', type=int, default=128)


args = parser.parse_args()


column_name = args.text_feature_col
data_dir = args.data_dir
w2v_word_count = args.w2v_word_count
learning_rate = args.learning_rate
n_epochs = args.epoch
batch_size = args.batch_size


torch.manual_seed(1151)

 
 # Load the data and skip bad lines
df = pd.read_csv(data_dir, header=None, engine='python', encoding='utf-8', on_bad_lines='skip')


df.columns = ['datasetId',
'paperId',
'corpusId',
'title',
'abstract',
'year',
'referenceCount',
'citationCount',
'influentialCitationCount',
'publicationDate',
'authors',
'pdfUrl',
'titleText',
'abstractText',
'sectionHeaderList',
'sectionHeaderDict',
'introduction',
'method',
'result',
'conclusion',
'citationCountRange']


min_value = 0
max_value = 2


def is_integer(val):
    if val is None:
        return False
    try:
        int(val)
        return True
    except ValueError:
        return False
    
if torch.cuda.is_available():  
    device = torch.device('cuda:0')
   
else:  
    device = torch.device('cpu')



# Apply the range filter for a specific column

mask = df['citationCountRange'].apply(is_integer)

df = df[mask]

df['citationCountRange'] = pd.to_numeric(df['citationCountRange'], errors='coerce')

df = df[((df['citationCountRange'] >= min_value) & (df['citationCountRange'] <= max_value))]


class_combine_df_original = pd.DataFrame()



class_combine_df_original['body'] = df[column_name].astype(str)
class_combine_df_original['citation'] = df['citationCountRange'].astype(int)
class_combine_df_original['year'] = df['year'].astype(int)

#print(class_combine_df_original)



import contractions

class_combine_df = class_combine_df_original.copy()

# Convert all reviews into lowercase
class_combine_df['body'] = class_combine_df['body'].str.lower()

# Remove HTML
class_combine_df['body'] = class_combine_df['body'].str.replace(r'<[^<>]*>', '', regex=True)

# Remove URLs
class_combine_df['body'] = class_combine_df['body'].str.replace('http\S+|www.\S+', '', case=False)

# Perform contractions - perform before remove non-alphabetical since apostrophe will be removed
class_combine_df['body'] = class_combine_df['body'].apply(lambda x: contractions.fix(str(x)))

# Remove non-alphabetical characters
class_combine_df['body'] = class_combine_df['body'].str.replace('[^a-zA-Z]', ' ', regex=True)

# Remove extra spaces
class_combine_df['body'] = class_combine_df['body'].str.replace(r'\s+', ' ', regex=True).str.strip()


# Load the pretrained "word2vec-google-news-300" Word2Vec model
wv = api.load('word2vec-google-news-300')




all_dataset = class_combine_df["body"]


# Initialize an empty list to store word vectors
word2vec_feature = []

# Iterate through the words and get the vectors
count_out = 0
print("start word2vec")
for dataset_item in all_dataset:
    word_vector = []

    words = dataset_item.split(" ")
    count = 0
    for word in words:
        if word in wv:
            if(count == 0):
                word_vector = [wv[word]]
                
            else:
                word_vector = np.concatenate((word_vector, [wv[word]]), axis = 0)
            
            count = count + 1
        else:
            pass
        if(count == w2v_word_count):
            break

    if(count < w2v_word_count):
        if(count == 0):
            word_vector = [np.zeros(300)]
        else:
            word_vector = np.concatenate((word_vector, [np.zeros(300)]), axis = 0)
        count = count + 1

        for i in range(count, w2v_word_count):
            word_vector = np.concatenate((word_vector, [np.zeros(300)]), axis = 0)

      
    word2vec_feature.append(word_vector)

print("end word2vec")

#Split for word2vec
x_word2vec = word2vec_feature.copy()

y_word2vec = class_combine_df["citation"].copy()

additional_feature = class_combine_df["year"].copy()

class ReviewDataset(object):
    def __init__(self, data):
        self.features = data.features
        self.labels = data.labels
        self.additional_feature = data.additional_feature

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        additional_feature = self.additional_feature[index]
        return feature, label, additional_feature
    
    def __len__(self):
        return len(self.features)

x_word2vec_transform = ReviewDataset(pd.DataFrame(data = {"features": x_word2vec, "labels": y_word2vec, "additional_feature": additional_feature}).reset_index(drop = True))


x_word2vec_train, x_word2vec_test, y_word2vec_train, y_word2vec_test = train_test_split(x_word2vec_transform, y_word2vec, test_size = 0.2, random_state=1150)


valid_size = 0.2

# obtain training indices that will be used for validation
num_train = len(x_word2vec_train)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

word2vec_train_loader = torch.utils.data.DataLoader(x_word2vec_train, batch_size = batch_size, sampler=train_sampler)
word2vec_valid_loader = torch.utils.data.DataLoader(x_word2vec_train, batch_size = batch_size, sampler=valid_sampler)
word2vec_test_loader = torch.utils.data.DataLoader(x_word2vec_test, batch_size = batch_size)






class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.input_size = input_size
        self.output_size = output_size
        
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True) 
        self.fc = nn.Linear(hidden_size+1, output_size)
        
 
    def forward(self, x, additional_feature):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)

        out, _ = self.LSTM(x, (h0, c0))
        additional_feature = additional_feature.unsqueeze(-1)
        out = torch.cat((out[:, -1, :], additional_feature), dim=1)        
        out = self.fc(out)
        return out



n_input = 300
n_hidden = w2v_word_count
n_categories = 3
LSTM_model = LSTM(n_input, n_hidden, n_categories).to(device)


# number of epochs to train the model
n_epochs = n_epochs

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf # set initial "min" to infinity

# specify loss function (categorical cross-entropy)


# from sklearn.utils.class_weight import compute_class_weight
# class_weights = compute_class_weight('balanced', classes=np.unique(y_word2vec_train), y=y_word2vec_train)
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# criterion = nn.CrossEntropyLoss(weight=class_weights)

criterion = nn.CrossEntropyLoss()

# specify optimizer (Adam) and learning rate
optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=learning_rate)

print("start training")

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    LSTM_model.train() # prep model for training
    
    for data, target, additional_feature in word2vec_train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        data, target, additional_feature = data.to(device, dtype=torch.float), target.to(device), additional_feature.to(device)
        output = LSTM_model(data, additional_feature)
        # calculate the loss
        loss = criterion(output, target.long())
        
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    LSTM_model.eval() # prep model for evaluation
    for data, target, additional_feature in word2vec_valid_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        data, target, additional_feature = data.to(device, dtype=torch.float), target.to(device), additional_feature.to(device)
        output = LSTM_model(data, additional_feature)
        # calculate the loss
        loss = criterion(output, target.long())
        # update running validation loss 
        valid_loss += loss.item()*data.size(0)
        
    # print training/validation statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(word2vec_train_loader.dataset)
    valid_loss = valid_loss/len(word2vec_valid_loader.dataset)
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, 
        train_loss,
        valid_loss
        ))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(LSTM_model.state_dict(), 'model_lstm_'+column_name+'.pt')
        valid_loss_min = valid_loss


LSTM_model.load_state_dict(torch.load('model_lstm_'+column_name+'.pt'))
test_loader = torch.utils.data.DataLoader(x_word2vec_test, batch_size=1)

def predict(LSTM_model, dataloader):
    prediction_list = []
    for data, target, additional_feature in dataloader:
        data, target, additional_feature = data.to(device, dtype=torch.float), target.to(device), additional_feature.to(device)
        outputs = LSTM_model(data, additional_feature)
        _, predicted = torch.max(outputs.data, 1) 
        prediction_list.append(predicted.cpu().numpy())
    return prediction_list

predictions = predict(LSTM_model, test_loader)
y_word2vec_mlp_pred = np.array(predictions)

accuracy_word2vec = accuracy_score(y_word2vec_test, y_word2vec_mlp_pred)
precision_word2vec = precision_score(y_word2vec_test, y_word2vec_mlp_pred, average='weighted')
recall_word2vec = recall_score(y_word2vec_test, y_word2vec_mlp_pred, average='weighted')
f1_word2vec = f1_score(y_word2vec_test, y_word2vec_mlp_pred, average='weighted')

print(accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec)
