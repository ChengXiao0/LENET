# LENET数字识别小项目
本项目是一个关于数字识别的小项目，麻雀虽小五脏俱全
1. train.py 训练
2. infer.py 利用opencv打开摄像头进行实时推理
3. ckp/ 保存模型的文件夹
4. dataset/mydata.py 自己定义的数据集
5. model/lenet.py 定义模型
## 使用指南
- 本例中作为数字识别的应用,每个样本的label在其文件名中,推荐将文件名按照label.index.jpg的样式来命名.
``` 1.12.jpg #指的是图片的label是1,是第12张```
- 当改变类别时,也应对应修改model/lenet.py内模型最后一层的输出,本例中含有8类数据,故:```self.f5 = nn.Linear(84, 8)```
- 注意样本的位置```dat = mydata("path")```
- 保存的条件的位置.eg:
```
if acc > 0.85:
        torch.save(net.state_dict(), f'/home/xiaoxiao/PycharmProjects/imgResize/ckp/model{epoch}_{acc}.pth')
```
- 代码中设定了计算准确率的,其中487为训练总图片数,请修改
```
acc = tR / 487
print("acc = ", tR / 487)
```
- 选定训练设备,batch,epoch
```
device = torch.device("cuda:0") #没有可用GPU请修改为CPU
batch = 16
epochs = 80
```
- 训练, 运行train.py即可
- 预测, 在infer.py中修改load的模型后,运行即可