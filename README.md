# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:
### Register Number:
```python
class Neuralnet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(1,8)
    self.fc2=nn.Linear(8,10)
    self.fc3=nn.Linear(10,1)
    self.relu=nn.ReLU()
    self.history={'loss':[]}
  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x



# Initialize the Model, Loss Function, and Optimizer
def train_model(Aadithyan_brain=_brain,x_train,y_train,criterion,optimizer,epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss=criterion(Aadithyan_brain(x_train),y_train)
    loss.backward()
    optimizer.step()
    Aadithyan_brain.history['loss'].append(loss.item())
    if epoch%200==0:
      print(f'epoch:[{epoch}/{epochs}], loss:{loss.item():.6f}')



```
## Dataset Information

<img width="549" height="685" alt="image" src="https://github.com/user-attachments/assets/8c0ca5b3-772b-4f23-89f3-b6a362e578c4" />
## OUTPUT

### Training Loss Vs Iteration Plot

<img width="866" height="566" alt="Screenshot 2025-09-16 112358" src="https://github.com/user-attachments/assets/52a1e0bd-ca84-45f5-9108-feb11204164c" />

### New Sample Data Prediction

<img width="1188" height="151" alt="Screenshot 2025-09-16 112630" src="https://github.com/user-attachments/assets/e6af2356-a1aa-4aba-89fb-7a7f9f59450a" />

## RESULT
The program to develop a neural network regression model for the given dataset has been executed successively
