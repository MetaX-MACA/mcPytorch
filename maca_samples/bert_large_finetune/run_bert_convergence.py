import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

import logging
torch.manual_seed(0)
torch.cuda.manual_seed(0)
log = logging.getLogger()
log.setLevel(logging.INFO)

class Config():
    train_data = '/netapp/pytorch/golden/bert_convergence/train_dataset.npy' # 训练集
    predict_data = '/netapp/pytorch/golden/bert_convergence/eval_dataset.npy' # 测试集
    result_data_save = 'result/submission.csv' # 预测结果
    device = 'cpu' # 训练驱动

    model_path = '/netapp/pytorch/golden/bert_large_uncased/' # 预训练模型
    # model_path = './result/model/epoch-2/' # 预训练模型
    model_save_path = 'result/model/' # 保存模型

    tokenizer = None # 预训练模型的tokenizer

    # 数据标签
    label_dict = {'晨会早报': 0, '宏观研报': 1, '策略研报': 2, '行业研报': 3, '公司研报': 4, '基金研报': 5, '债券研报': 6, '金融工程': 7, '其他研报': 8, '个股研报': 9}
    num_labels = len(label_dict) # 标签数量

    max_seq_len = 128 # 最大句子长度
    test_size = 0.15 # 校验集大小
    random_seed = 42 # 随机种子
    batch_size = 8 # 训练数据批大小
    val_batch_size = 8 # 校验/预测批大小
    epochs = 50 # 训练次数
    learning_rate = 1e-5 # 学习率
    l2_weight_decay = 0.05
    print_log = 20 # 日志打印步骤

config = Config()
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 自定义dataset
class MyDataset(Dataset):
    def __init__(self, config: Config, data: list, label: list = None):
        self.data = data
        self.tokenizer = config.tokenizer
        self.max_seq_len = config.max_seq_len
        self.len = len(data)
        self.label = label

    def __getitem__(self, idx):
        text = self.data[idx]
        # tokenizer
        inputs = self.tokenizer.encode_plus(text, return_token_type_ids=True, return_attention_mask=True,
                                            max_length=self.max_seq_len, padding='max_length', truncation=True)

        # 打包预处理结果
        result = {'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                  'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
                  'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)}
        if self.label is not None:
            result['labels'] = torch.tensor([self.label[idx]], dtype=torch.long)
        # 返回
        return result

    def __len__(self):
        return self.len

train_data = pd.DataFrame(list(np.load(config.train_data, allow_pickle=True)))
train_data.head(5)

tokenizer = AutoTokenizer.from_pretrained(config.model_path)
print(tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(config.model_path, num_labels=config.num_labels)
print(model)
config.tokenizer = tokenizer

# 拼接生成最终的文本
train_data['text'] = train_data['header'] + '[SEP]' + train_data['title'] + '[SEP]' + train_data['paragraph'] + '[SEP]' + train_data['footer']
# 切分数据
X_train, X_val, y_train, y_val = train_test_split(train_data['text'].tolist(), train_data['label'].tolist(),
                                                          test_size=config.test_size,
                                                          random_state=config.random_seed)
# 构建数据
train_dataloader = DataLoader(MyDataset(config, X_train, y_train), batch_size=config.batch_size, shuffle=True)
val_dataloader = DataLoader(MyDataset(config, X_val, y_val), batch_size=config.val_batch_size, shuffle=True)

# 校验方法
def val(model, val_dataloader: DataLoader):
    model.eval()
    total_acc, total_f1, total_loss, test_num_batch = 0., 0., 0., 0
    for iter_id, batch in enumerate(val_dataloader):
        # 转GPU
        batch_cuda = {item: value.to(config.device) for item, value in batch.items()}
        # 模型计算
        output = model(**batch_cuda)
        # 获取结果
        loss = output[0]
        logits = torch.argmax(output[1], dim=1)

        y_pred = [[i] for i in logits.cpu().detach().numpy()]
        y_true = batch_cuda['labels'].cpu().detach().numpy()
        # 计算指标
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        total_loss += loss.item()
        total_acc += acc
        total_f1 += f1
        test_num_batch += 1

    return total_loss/test_num_batch, total_acc/test_num_batch, total_f1/test_num_batch

# 训练方法
def train(model, config: Config, train_dataloader: DataLoader, val_dataloader: DataLoader):
    # 模型写入GPU
    model.to(config.device)

    # 获取BERT模型的所有可训练参数
    params = list(model.named_parameters())
    # 对除了bias和LayerNorm层的所有参数应用L2正则化
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
         'weight_decay': config.l2_weight_decay},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    # 创建优化器并使用正则化更新模型参数
    opt = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    # 梯度衰减
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, len(train_dataloader) * config.epochs)

    # 遍历训练
    best_f1 = 0
    for epoch in range(config.epochs):
        total_acc, total_f1, total_loss, train_num_batch = 0., 0., 0., 0
        model.train()
        zero_step = 0
        for iter_id, batch in enumerate(train_dataloader):
            # 数据写入GPU
            batch_cuda = {item: value.to(config.device) for item, value in batch.items()}
            # 模型计算
            output = model(**batch_cuda)
            # 获取结果
            loss = output[0]
            logits = torch.argmax(output[1], dim=1)

            y_pred = [[i] for i in logits.cpu().detach().numpy()]
            y_true = batch_cuda['labels'].cpu().detach().numpy()

            # 计算指标
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            total_loss += loss.item()
            total_acc += acc
            total_f1 += f1

            # 反向传播，更新参数
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            # 打印
            if iter_id % config.print_log == 0:
                logging.info('epoch:{}, iter_id:{}, loss:{}, acc:{}, f1:{}'.format(epoch, iter_id, loss.item(), acc, f1))

            train_num_batch += 1
        # 校验操作
        val_loss, val_acc, val_f1 = val(model, val_dataloader)
        if val_f1 > best_f1:
            best_f1 = val_f1
        # 保存当前epoch模型参数
        config.tokenizer.save_pretrained(config.model_save_path + f"/epoch-{epoch}")
        model.save_pretrained(config.model_save_path + f"/epoch-{epoch}")
        logging.info('-' * 15+str(epoch)+'-' * 15)
        logging.info('avg_train_loss:{}, avg_train_acc:{}, avg_train_f1:{}'.format(total_loss/train_num_batch, total_acc/train_num_batch, total_f1/train_num_batch))
        logging.info('val_loss:{}, val_acc:{}, val_f1:{}, best_f1:{}'.format(val_loss, val_acc, val_f1, best_f1))
        logging.info('-' * 30)

    # 保存最终模型
    config.tokenizer.save_pretrained(config.model_save_path)
    model.save_pretrained(config.model_save_path)

# 开始训练
train(model, config, train_dataloader, val_dataloader)
print('train done.')