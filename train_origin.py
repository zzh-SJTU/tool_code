from resnet_cifar import ResNet50
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
from torchvision import transforms
Batchsize=128
train_data = datasets.SVHN(root='SVHN/',split='train',transform=ToTensor(), download=True)
test_data = datasets.SVHN(root='SVHN/',split='test',transform=ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=Batchsize,
                                          shuffle=True,
                                          )
test_loader=torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=Batchsize,
                                          shuffle=True,
                                          )
resnet34 = ResNet50(num_classes=10)
#resnet34.conv1=nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
resnet34 = resnet34.to('cuda')
Epoch=30
loss_fn=nn.CrossEntropyLoss()
lr=0.01
opt=torch.optim.SGD(resnet34.parameters(),lr)
for epoch in range(Epoch):
    counter=0
    loss_epoch=0
    for x,y in train_loader:
        x=x.cuda()
        y=y.cuda()
        out=resnet34(x)
        loss=loss_fn(out,y)
        loss_epoch=loss_epoch+loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    print('loss', loss_epoch.item()/len(train_loader))
    total=0
    correct=0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images=images.cuda()
        # calculate outputs by running images through the network
            outputs = resnet34(images)
            outputs = outputs.to('cpu')
        # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(correct/total)
print('Finish Training')
torch.save(resnet34.state_dict(), 'resnet50_SVHN_weights.pth')