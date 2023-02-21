from MAP_DNN.losses import loss_fn
from MAP_DNN.train import train_and_valid
from MAP_DNN.model import *
from MAP_DNN.dataloader import load_data

file = pd.read_csv("example/example.csv")
print(file.head())
target_str = "Post"
pre_str = "Pre"
ratio = np.sum(file[pre_str]) / np.sum(file[target_str])

#hyperparameters
L1_rate = 1
n_epochs = 100
learning_rate = 0.001

model=Net()
init_weights(model)

train_data_loader,valid_data_loader,test_data_loader=load_data(file,target_str,pre_str)
model=train_and_valid(model,n_epochs,train_data_loader,valid_data_loader,ratio,loss_fn,0.001)

path="example.pt"
torch.save(model.state_dict(),path)
