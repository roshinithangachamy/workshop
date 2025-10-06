## workshop:Binary Classification with Neural Networks on the Census Income Dataset

## Aim:
To build a binary classification model using PyTorch to predict whether an individual earns more than $50,000 annually.

## Code:
```
import torch
import torch.nn as nn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
%matplotlib inline

df = pd.read_csv('/content/income.csv')
print(len(df))
df.head()

df['label'].value_counts()

df.columns
cat_cols = ['sex', 'education', 'marital-status', 'workclass', 'occupation']
cont_cols = ['age', 'hours-per-week']
y_col = ['label']

for col in cat_cols:
    df[col] = df[col].astype('category')

df = shuffle(df, random_state=101)
df.reset_index(drop=True, inplace=True)
df.head()

cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)

# CODE HERE

cats = np.stack([df[col].cat.codes.values for col in cat_cols], axis=1)

# RUN THIS CODE TO COMPARE RESULTS
cats[:5]

# CODE HERE
cats = torch.tensor(cats, dtype=torch.int64)
print(cats[:5])

# CODE HERE

conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts[:5]

# CODE HERE
# RUN THIS CODE TO COMPARE RESULTS
conts = torch.tensor(conts, dtype=torch.float32)
print(conts.dtype)

y = torch.tensor(df[y_col].values, dtype=torch.int64).flatten()
# CODE HERE
b = 30000 # suggested batch size
t = 5000  # suggested test size

cat_train = cats[:b-t]
cat_test = cats[b-t:]
con_train = conts[:b-t]
con_test = conts[b-t:]
y_train = y[:b-t]
y_test = y[b-t:]

class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        # Call the parent __init__
        super().__init__()
        
        # Set up the embedding, dropout, and batch normalization layer attributes
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        # Assign a variable to hold a list of layers
        layerlist = []
        
        # Assign a variable to store the number of embedding and continuous layers
        n_emb = sum((nf for ni,nf in emb_szs))
        n_in = n_emb + n_cont
        
        # Iterate through the passed-in "layers" parameter (ie, [200,100]) to build a list of layers
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
        
        # Convert the list of layers into an attribute
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        # Extract embedding values from the incoming categorical data
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        # Perform an initial dropout on the embeddings
        x = self.emb_drop(x)
        
        # Normalize the incoming continuous data
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        
        # Set up model layers
        x = self.layers(x)
        return x
# CODE HERE
torch.manual_seed(33)

# CODE HERE
model = TabularModel(emb_szs=emb_szs, n_cont=len(cont_cols), out_sz=2, layers=[50], p=0.4)
# RUN THIS CODE TO COMPARE RESULTS
model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
import time
start_time = time.time()

epochs = 300
losses = []

for i in range(epochs):
    i+=1
    y_pred = model(cat_train, con_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    
    if i%25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {time.time() - start_time:.0f} seconds')

losses = [l.item() if torch.is_tensor(l) else l for l in losses]
plt.plot(np.array(losses, dtype=float))
plt.ylabel("Cross Entropy Loss")
plt.xlabel("Epoch")
plt.title("Model Training Loss")
plt.show()

with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = criterion(y_val, y_test)

# RUN THIS CODE TO COMPARE RESULTS
print(f'CE Loss: {loss:.8f}')

correct = 0
for i in range(len(y_test)):
    if torch.argmax(y_val[i]).item() == y_test[i].item():
        correct += 1
print(f"{correct} out of {len(y_test)} = {100 * correct / len(y_test):.2f}% correct")

# WRITE YOUR CODE HERE:
def predict_income(model, encoders, cont_inputs, cat_inputs):
    model.eval()
    cat_tensor = torch.tensor([cat_inputs], dtype=torch.int64)
    cont_tensor = torch.tensor([cont_inputs], dtype=torch.float32)
    with torch.no_grad():
        output = model(cat_tensor, cont_tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred
# Example mappings 
sex_d = {'Female': 0, 'Male': 1}
education_d = {
    '3': 0, '4': 1, '5': 2, '6': 3, '6': 4, '8': 5, '12': 6,
    'HS-grad': 7, 'Some-college': 8, 'Assoc-voc': 9, 'Assoc-acdm': 10, 'Bachelors': 11,
    'Masters': 12, 'Prof-school': 13, 'Doctorate': 14
}
marital_d = {'Divorced': 0, 'Married': 1, 'Married-spouse-absent': 2, 'Never-married': 3, 'Separated': 4, 'Widowed': 5}
workclass_d = {'Federal-gov': 0, 'Local-gov': 1, 'Private': 2, 'Self-emp': 3, 'State-gov': 4}
occupation_d = {'Adm-clerical': 0, 'Craft-repair': 1, 'Farming-fishing': 2, 'Handlers-cleaners': 3,
                'Machine-op-inspct': 4, 'Other-service': 5, 'Prof-specialty': 6, 'Protective-serv': 7,
                'Sales': 8, 'Tech-support': 9, 'Transport-moving': 10}

# Get inputs from user, convert and encode
age = int(input("What is the person's age? (18-90) "))
sex = sex_d[input("What is the person's sex? (Male/Female) ").capitalize()]
education = education_d[input("What is the person's education level? ").strip()]
marital = marital_d[input("What is the person's marital status? ").strip()]
workclass = workclass_d[input("What is the person's workclass? ").strip()]
occupation = occupation_d[input("What is the person's occupation? ").strip()]
hours_per_week = int(input("How many hours/week are worked? (20-90) "))

cat_inputs = [sex, education, marital, workclass, occupation]
cont_inputs = [age, hours_per_week]

predicted_label = predict_income(model, None, cont_inputs, cat_inputs)

print(f"\nThe predicted label is {predicted_label}")
```

## Output:
<img width="1101" height="263" alt="image" src="https://github.com/user-attachments/assets/e85338f8-a9d9-47fc-83ee-85761d54628c" />

<img width="192" height="211" alt="image" src="https://github.com/user-attachments/assets/7dcd5018-eec6-4bb5-a598-9bccb7c65db6" />

<img width="767" height="83" alt="image" src="https://github.com/user-attachments/assets/a86045e0-9ec7-489a-9127-6b4ab4ee0726" />

<img width="278" height="88" alt="image" src="https://github.com/user-attachments/assets/2219cd03-ef0d-45f5-99ad-6148e0e1521c" />

<img width="1093" height="223" alt="image" src="https://github.com/user-attachments/assets/03ec0200-999f-4ccd-ad0c-02dac5d530ee" />

<img width="432" height="52" alt="image" src="https://github.com/user-attachments/assets/c30bd1c9-25fe-4e37-ac16-df061b8aaef8" />

<img width="503" height="123" alt="image" src="https://github.com/user-attachments/assets/1648700b-de05-4652-a179-412c871cc957" />

<img width="353" height="136" alt="image" src="https://github.com/user-attachments/assets/694468a5-acea-49aa-bfaf-3413c5179922" />

<img width="211" height="119" alt="image" src="https://github.com/user-attachments/assets/a1f4d218-8aea-4c62-b47a-538d8673914c" />

<img width="828" height="375" alt="image" src="https://github.com/user-attachments/assets/430e46c0-b5ee-4d32-984a-3a15318212e4" />

<img width="342" height="321" alt="image" src="https://github.com/user-attachments/assets/7bf7c388-e040-4444-ac90-2dd2a037ac69" />

<img width="686" height="522" alt="image" src="https://github.com/user-attachments/assets/3434a05d-68a1-4cf7-b5dc-5b63f49b5694" />

<img width="228" height="59" alt="image" src="https://github.com/user-attachments/assets/f9495cec-8591-46f7-af41-3e9e84a4d7d8" />

<img width="606" height="225" alt="image" src="https://github.com/user-attachments/assets/c8ef6820-1cc4-488c-ba06-75dd0bbf08a0" />


## Result:
Thus building a binary classification model using PyTorch to predict whether an individual earns more than $50,000 annually has been done.
