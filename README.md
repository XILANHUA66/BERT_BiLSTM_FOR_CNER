# BERT_BiLSTM_FOR_CNER
这是一个基于预训练模型的用来完成中文命名实体识别任务的模型<br>
整体结构：BERT+BiLSTM<br>
小模型train可以直接在kaggle平台训练，在输出文件final_model是保留的模型配置和权重<br>
final_model和test.py放到统一目录运行下即可<br>
本项目小模型是论文的简化版<br>
>https://github.com/StanleySun233/HREB-CRF.git
><br>Bert+BiLSTM+Mega+CRF

只是用来满足毕业设计需求,AI辅助代码,痕迹比较重<br>
## 数据集
MSRA推出的关于中文命名实体识别的数据集
```python
dataset = load_dataset("PassbyGrocer/msra-ner")
```
## Bert
google的面向中文文本的pretrain的bert
```python
MODEL_PATH = "bert-base-chinese"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertBiLSTMForNER.from_pretrained(MODEL_PATH, num_labels=len(label_list))
```
## 模型的下载
如果想要获得已经训练好的权重<br>
>https://huggingface.co/XILANHUA11/BERT_BiLSTM_FOR_CNER

下载final_model,并可以用test.py做调用
