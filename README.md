# CS6910-Assignment-3 (Sequence to Sequence Model for Transliteration)
# Author: Abhishek Koushal(CS23M007)

# Report
The Link of report for this assignment https://wandb.ai/cs23m007/dl-assignment-3/reports/Assignment-3--Vmlldzo3NjEwNzgz

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
____________________________________________________________________________________________________________________________________________________________________________
                        parser = argparse.ArgumentParser(description='Execute the model and calculate the accuracy')
                        parser.add_argument('-wp', '--wandb_project', type=str, help='wandb project name', default='cs6910_assignment3')
                        parser.add_argument('-we', '--wandb_entity', type=str, help='wandb entity', default='cs22m029')
                        parser.add_argument('-es', '--emb_size', type=int, help='embedding size', default=256)
                        parser.add_argument('-nle', '--num_layers_encoder', type=int, help='number of layers in encoder', default=2)
                        parser.add_argument('-nld', '--num_layers_decoder', type=int, help='number of layers in decoder', default=2)
                        parser.add_argument('-hs', '--hidden_size', type=int, help='hidden size', default=256)
                        parser.add_argument('-bs', '--batch_size', type=int, help='batch size', default=32)
                        parser.add_argument('-ep', '--epochs', type=int, help='epochs', default=5)
                        parser.add_argument('-ct', '--cell_type', type=str, help='Cell type', default="LSTM")
                        parser.add_argument('-bdir', '--bidirectional', type=bool, help='bidirectional', default=False)
                        parser.add_argument('-drop', '--dropout', type=float, help='dropout', default=0.2)
____________________________________________________________________________________________________________________________________________________________________________

The train.py file is responsible for training the transliteration model. It utilizes a sequence-to-sequence architecture, which consists of an encoder and a decoder. The encoder processes the input sequence (source language) and encodes it into a fixed-length vector representation. The decoder takes this representation and generates the output sequence (target language).

There are two versions of the model implemented in the train.py file: vanilla and attention. The vanilla model is the basic implementation of the sequence-to-sequence model, while the attention model incorporates attention mechanisms to improve the model's performance.
