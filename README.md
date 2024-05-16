# CS6910-Assignment-3 (Sequence to Sequence Model for Transliteration)
# Author: Abhishek Koushal(CS23M007)

# Report
The Link of report for this assignment https://wandb.ai/cs23m007/dl-assignment-3/reports/Assignment-3--Vmlldzo3NjEwNzgz

## Setup

**Note:** It is recommended to create a new python virtual environment before installing dependencies.

```
pip install requirements.txt
```

To Run Model

```
python train.py --epochs 10 --batch_size 16
``` 


# Description of Files:
**File 1: "predictions_vanilla.csv"**

Description:
The "predictions_vanilla.csv" file contains all the test data predictions made by the best vanilla model. These predictions are the result of training and evaluation using vanilla sequence to sequence model (seq2seq).

**File 2: "predictions_attention.csv"**

Description:
The "predictions_attention.csv" file consists of all the test data predictions made by the best attention model. These predictions are the result of training and evaluation using vanilla sequence to sequence model (seq2seqWithAttention).

**File 3: "dl-assignment-3-without-attension.ipynb"**

Description:
The "dl-assignment-3-without-attension.ipynb" file is a one time execution of the Vanilla transliteration sequence to sequence model on Google colab.

**File 4: "dl-assignment-3-with-attension.ipynb"**

Description:
The "dl-assignment-3-with-attension.ipynb" file is a one time execution of the Vanilla transliteration sequence to sequence model (flag=False) and attention model (flag=True) on Kaggle.

Add the aksharantar-sampled dataset from Kaggle and in notebook options select accelerator as GPU to execute the file.


**File 5: "train.py"**

Description:
The train.py file contains the implementation of both vanilla and attention-based models for transliteration using a sequence-to-sequence architecture. 

To execute the file, follow these steps:
1. Ensure that the aksharantar_sampled file is present in the same folder as the train.py file.
2. Provide the necessary values for all the required arguments while executing the file directly from the terminal. Alternatively, if no arguments are provided, the file will use the default values, which are set to the best parameters. The arguments that can be used are following :

### Arguments

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | YOUR_WANDB_PROJECT_NAME | Wandb project name |
| `-we`, `--wandb_entity` | YOUR_WANDB_ENTITY | wandb entity | 
| `-es`, `--emb_size` | 256 | embedding size, choice:[16, 32, 64, 256] |
| `-nle`, `--num_layers_encoder` | 3 | number of layers in encoder, choices:[1, 2, 3] | 
| `-nld`, `--num_layers_decoder` | 3 | number of layers in decoder, choices:[1, 2, 3] | 
| `-hs`, `--hidden_size` | 64 | hidden size, choices:  [16, 32, 64, 256] | 
| `-bs`, `--batch_size` | 32 | batch size, choices: [32, 64, 128] | 
| `-ep`, `--epochs` | 5 | Number of epochs to train neural network.[5, 10, 15, 20] | 
| `-ct`, `--cell_type` | "LSTM" | Cell type, choices:  ["RNN", "GRU", "LSTM"] | 
| `-bdir`, `--bidirectional` | False | choices:  [True, False] | 
| `-drop`, `--dropout` | 0.2 | dropout value, choices: [0,0.2,0.3] | 
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters, choices: [0.001,0.0001,0.0003,0.0005] | 
| `-at`, `--attention` | 0.001 | Simple seqtoseq model or with attention mechanism, choices: [True, False] | 
____________________________________________________________________________________________________________________________________________________________________________

The train.py file is responsible for training the transliteration model. It utilizes a sequence-to-sequence architecture, which consists of an encoder and a decoder. The encoder processes the input sequence (source language) and encodes it into a fixed-length vector representation. The decoder takes this representation and generates the output sequence (target language).

There are two versions of the model implemented in the train.py file: vanilla and attention. The vanilla model is the basic implementation of the sequence-to-sequence model, while the attention model incorporates attention mechanisms to improve the model's performance.
