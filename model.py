import torch

from torch.nn.modules.loss import MSELoss

import torch.nn.functional as F





class RegressionTrain(torch.nn.Module):
    '''
    '''

    def __init__(self, model):
        '''
        '''

        # initialize the module using super() constructor
        super(RegressionTrain, self).__init__()
        # assign the architectures
        # 在我们的版本中model不被定义为属性而是在主程序中手动调用
        self.model = model
        # assign the weights for each task
        # weights: 一个 torch.nn.Parameter 对象，用于表示每个任务的权重。
        # model.n_tasks 指定了任务的数量，初始化为每个任务的权重为1。
        # weights函数就是一个针对每个任务的权重向量 torch.ones(model.n_tasks)表示生成一个长度为n_tasks的全为1的向量 就是初始向量
        self.weights = torch.nn.Parameter(torch.ones(model.n_tasks).float())
        
        # loss function
        self.mse_loss = MSELoss()

    
    def forward(self, x, ts):
        # x: 输入数据，形状为 [B, D]，其中 B 是批次大小，D 是输入特征维度。
        # ts: 目标数据，形状为 [B, n_tasks, D']，每个任务对应一个目标值 也就是真实标签，用于和预测标签对比计算真实值

        
        B, n_tasks = ts.shape[:2]
        ys = self.model(x)   # ys是预测结果
        
        # check if the number of tasks is equal to this size
        # assert 是 Python 的一个内置关键字，用于检查条件是否为真。
        # 如果条件为 True，assert 不做任何操作；如果条件为 False，则抛出 AssertionError 异常。
        assert(ys.size()[1] == n_tasks)
        task_loss = []
        for i in range(n_tasks):
            task_loss.append( self.mse_loss(ys[:,i,:], ts[:,i,:]) )
        task_loss = torch.stack(task_loss)

        return task_loss


    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()



class RegressionModel(torch.nn.Module):
    '''
    '''

    def __init__(self, n_tasks):
        '''
        Constructor of the architecture.
        Input:
            n_tasks: number of tasks to solve ($T$ in the paper)
        '''

        # initialize the module using super() constructor
        super(RegressionModel, self).__init__()
        
        # number of tasks to solve
        self.n_tasks = n_tasks
        # fully connected layers
        self.l1 = torch.nn.Linear(250, 100)
        self.l2 = torch.nn.Linear(100, 100)
        self.l3 = torch.nn.Linear(100, 100)
        self.l4 = torch.nn.Linear(100, 100)
        # branches for each task
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), torch.nn.Linear(100, 100))

    
    def forward(self, x):
        # forward pass through the common fully connected layers
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))

        # forward pass through each output layer
        outs = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outs.append(layer(h))

        return torch.stack(outs, dim=1)


    def get_last_shared_layer(self):
        return self.l4

        
