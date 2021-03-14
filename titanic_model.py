from collections import namedtuple
import copy
import torch as t
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, SubsetRandomSampler


class TitanicDataset(Dataset):
    def __init__(self, data_x, data_y, mode=''):
        self.data_x = data_x
        self.mode = mode
        if self.mode != 'test':
            self.data_y = data_y.values.tolist()

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        if self.mode != 'test':
            return t.FloatTensor(self.data_x.iloc[index]), t.FloatTensor(self.data_y[index])
        else:
            return t.FloatTensor(self.data_x.iloc[index])


def train_model(model, train_loader, val_loader, loss, optimizer, scheduler, num_epochs, threshold):
    loss_history = []
    train_history = []
    val_history = []
    for epoch in range(num_epochs):
        model.train()

        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        for i_step, (x, y) in enumerate(train_loader):
            prediction = model(x)
            loss_value = loss(prediction, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            winners = prediction >= threshold
            correct_samples += t.sum(winners == y)
            total_samples += y.shape[0]

            loss_accum += loss_value

        ave_loss = loss_accum / len(train_loader)
        train_accuracy = float(correct_samples) / total_samples

        if val_loader:
            val_accuracy = compute_accuracy(model, val_loader, threshold)
            val_history.append(val_accuracy)
            print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        scheduler.step(ave_loss)

    if val_loader:
        return loss_history, train_history, val_history
    else:
        return loss_history, train_history


def compute_accuracy(model, loader, threshold):
    model.eval()

    corrects = 0
    num_samples = 0
    for i_step, (x, y) in enumerate(loader):
        winners = model(x) >= threshold
        corrects += t.sum(winners == y)
        num_samples += y.shape[0]
    return corrects / num_samples


def seed_init_fn(seed):
   np.random.seed(seed)
   t.manual_seed(seed)


seed = 1643
seed_init_fn(seed)

train_x = pd.read_csv('/Users/kamyshnikovy/PycharmProjects/pythonProject/kaggle/titanic_train_x.csv')
train_y = pd.read_csv('/Users/kamyshnikovy/PycharmProjects/pythonProject/kaggle/titanic_train_y.csv')
train = TitanicDataset(train_x, train_y)

test = pd.read_csv('/Users/kamyshnikovy/PycharmProjects/pythonProject/kaggle/titanic_test.csv')
passenger_id = test[['PassengerId']]
test_to_model = test.drop('PassengerId', axis=1)

data_size = len(train)
validation_split = .15
split = int(np.floor(validation_split * data_size))
indices = list(range(data_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

Hyperparams = namedtuple("Hyperparams", ['learning_rate', 'threshold', 'reg', 'batch_size', 'anneal_coeff', 'patience'])
RunResult = namedtuple("RunResult", ['model', 'train_history', 'val_history', 'final_val_accuracy'])
run_record = {}

loss = nn.BCELoss()
epoch_num = 20
learning_rates = [1.4875]
regularisation = [0.075]
batch_size = [758]
anneal_coeff = [0.665]
threshold = [0.39]
patience = [0]

model_S = nn.Sequential(
          nn.BatchNorm1d(11),
          nn.Linear(11, 4),
          nn.ReLU(),
          nn.BatchNorm1d(4),
          nn.Linear(4, 1),
          nn.Sigmoid()
)

for th in threshold:
    for reg in regularisation:
        for coeff in anneal_coeff:
            for size in batch_size:
                for pat in patience:
                    for lr in learning_rates:
                        model = copy.deepcopy(model_S)

                        print('LR:', lr, '   reg:', reg, '   batch_size:', size, '   coeff:', coeff, '   threshold:', th, '   patience:', pat)
                        train_loader = t.utils.data.DataLoader(train, batch_size=size, sampler=train_sampler, worker_init_fn=seed_init_fn(seed))
                        val_loader = t.utils.data.DataLoader(train, batch_size=size, sampler=val_sampler, worker_init_fn=seed_init_fn(seed))
                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
                        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=coeff, patience=pat)
                        loss_history, train_history, val_history = train_model(model, train_loader, val_loader, loss, optimizer, scheduler, epoch_num, th)
                        plt.plot(loss_history)
                        plt.show()
                        plt.plot(train_history)
                        plt.plot(val_history)
                        plt.show()
                        run_record[Hyperparams(lr, th, reg, size, coeff, pat)] = RunResult(model, train_history, val_history, val_history[-1])

best_val_accuracy = None
best_hyperparams = None
best_run = None
final_train_accuracy = None

for hyperparams, run_result in run_record.items():
    if best_val_accuracy is None or best_val_accuracy < run_result.final_val_accuracy:
        best_val_accuracy = run_result.final_val_accuracy
        final_train_accuracy = run_result.train_history[-1]
        best_hyperparams = hyperparams
        best_run = run_result
print("Best validation accuracy: %4.2f, Best train accuracy: %4.2f,  best hyperparams: %s" % (best_val_accuracy, final_train_accuracy, best_hyperparams))

lr, th, reg, size, coeff, pat = best_hyperparams
train_loader = t.utils.data.DataLoader(train, batch_size=size, worker_init_fn=seed_init_fn(seed))
optimizer = optim.Adam(model_S.parameters(), lr=lr, weight_decay=reg)
scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=coeff, patience=pat)
loss_history, train_history = train_model(model_S, train_loader, None, loss, optimizer, scheduler, epoch_num, th)

model_S.eval()

test_dataset = TitanicDataset(test_to_model, None, mode='test')
test_loader = t.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))

model_probs = model_S(next(iter(test_loader)))
model_preds = model_probs >= th
df_model_preds = pd.DataFrame(model_preds)

test_predictions = df_model_preds.replace([True, False], [1, 0])
test_result = passenger_id.join(test_predictions)
test_result.columns = ['PassengerId', 'Survived']
test_result.to_csv('/Users/kamyshnikovy/PycharmProjects/pythonProject/kaggle/titanic_result.csv', index=False)

# Best validation accuracy: 0.80
# submission score 0.65789