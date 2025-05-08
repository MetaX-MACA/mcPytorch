# Instruction For Running Bert AMP

1. How to run?
Normally you can run below command to trigger the test
```
cd maca_samples/amp/
bash run_bert_amp.sh
```

Below are some useful options:

- \-\-offline
if you cannot download model from website during run test, you can download the model(bert-base-uncased) manually, then use this option.
Please note before running test, you need specific you model path by 
```
--weight_path=your-path/pytorch_model.bin
```

- \-\-save_log
This option would save a log file into your current direction.

