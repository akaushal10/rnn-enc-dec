
import os
import wandb
import torch
import random
import heapq
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import pandas as pd
import torch.optim as optim
import torch.nn.functional as Function
import csv
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt
import argparse

SYMBOL_BEGIN, SYMBOL_END, SYMBOL_UNKNOWN, SYMBOL_PADDING = 0, 1, 2, 3

INPUT_LABEL = "input"
TARGET_LABEL = "target"
DELIMETER = ","

RNN_KEY = "RNN"
GRU_KEY = "GRU"
LSTM_KEY = "LSTM"

INPUT_LANG_KEY = "input_lang"
OUTPUT_LANG_KEY = "output_lang"
SOURCE_LANG_KEY = "source_lang"
TARGET_LANG_KEY = "target_lang"

PAIRS_KEY = "pairs"
MAX_LEN_KEY = "max_len"

INPUT_LANG = "eng"
TARGET_LANG = "hin"

TRAIN_LABEL = "train"
TEST_LABEL = "test"
VALID_LABEL = "valid"

DEFAULT_PATH = "/kaggle/input/aksharantar-sampled/aksharantar_sampled"
TRAIN_DATASET_PATH = f"{DEFAULT_PATH}/{TARGET_LANG}/{TARGET_LANG}_{TRAIN_LABEL}.csv"
VALIDATION_DATASET_PATH = f"{DEFAULT_PATH}/{TARGET_LANG}/{TARGET_LANG}_{VALID_LABEL}.csv"
TEST_DATASET_PATH = f"{DEFAULT_PATH}/{TARGET_LANG}/{TARGET_LANG}_{TEST_LABEL}.csv"

PREDICTION_WITHOUT_ATTENSION_FILE_NAME = "predictions_vanilla.csv"
PREDICTION_WITH_ATTENSION_FILE_NAME = "predictions_attention.csv"

NADAM_KEY = "Nadam"

# Sweep param labels
EMBEDDING_SIZE_KEY = "embedding_size"
EPOCHS_KEY = "epochs"
ENCODER_LAYER_KEY = "encoder_layers"
DECODER_LAYER_KEY = "decoder_layers"
HIDDEN_LAYER_KEY = "hidden_layer"
IS_BIDIRECTIONAL_KEY = "bidirectional"
DROPOUT_KEY = "dropout"
CELL_TYPE_KEY = "cell_type"
LEARNING_RATE_KEY = "learning_rate"
BATCH_SIZE_KEY = "batch_size"

# wandb constants
WANDB_PROJECT_NAME="dl-assignment-3"
WANDB_ENTITY_NAME="cs23m007"
WANDB_FONT_FAMILY="MANGAL.TTF"

# wandb plot titles
TRAIN_ACCURACY_TITLE = "train_acc"
VALIDATION_ACCURACY_TITLE = "val_acc"
VALIDATION_ACCURACY_CHAR_TITLE = "val_acc_char"
VALIDATION_ACCURACY_WORD_TITLE = "val_acc_word"
CORRECTLY_PREDICTED_TITLE = "correctly_predicted"

TEST_ACCURACY_TITLE = "test_acc"
TRAIN_LOSS_TITLE = "train_loss"
VALIDATION_LOSS_TITLE = "val_loss"
TEST_LOSS_TITLE = "test_loss"

CSV_COLUMN_ACTUAL_X = "Actual_X"
CSV_COLUMN_ACTUAL_Y = "Actual_Y"
CSV_COLUMN_PREDICETED_Y = "Predicted_Y"

INPUT_INDEX_KEY = "input_index"
OUTPUT_INDEX_KEY = "output_index"

best_params = {
    EMBEDDING_SIZE_KEY :256,
    EPOCHS_KEY :5,
    ENCODER_LAYER_KEY :2,
    DECODER_LAYER_KEY :2,
    HIDDEN_LAYER_KEY :256,
    IS_BIDIRECTIONAL_KEY :False,
    DROPOUT_KEY :0.2,
    CELL_TYPE_KEY :LSTM_KEY,
    BATCH_SIZE_KEY : 32,
    LEARNING_RATE_KEY: 0.001

}


parser=argparse.ArgumentParser()

parser.add_argument('-wp','--wandb_project',help='wandb project name',type=str, default=WANDB_PROJECT_NAME)
parser.add_argument('-we','--wandb_entity', help='wandb entity name',type=str, default=WANDB_ENTITY_NAME)
parser.add_argument('-e','--epochs',help='epochs',choices=[5,10],type=int, default=10)
parser.add_argument('-b','--batch_size',help='batch sizes',choices=[32, 64, 128], type=int, default=128)
parser.add_argument('-lr','--learning_rate',help='learning rates', choices=[0.001,0.0001,0.0003,0.0005], type=float,default=1e-3)
parser.add_argument('-nle','--num_layers_en',help='number of layers in encoder',choices=[1,2,3],type=int, default=2)
parser.add_argument('-nld','--num_layers_dec',help='number of layers in decoder',choices=[1,2,3],type=int, default=3)
parser.add_argument('-sz','--hidden_size',help='hidden layer size',choices=[16, 32, 64, 256],type=int, default=512)
parser.add_argument('-il','--input_lang',help='input language', choices=[INPUT_LANG],type=str, default=INPUT_LANG)
parser.add_argument('-tl','--target_lang',help='target language',choices=[TARGET_LANG,'tel'],type=str, default=TARGET_LANG)
parser.add_argument('-ct','--cell_type',help='cell type',choices=[RNN_KEY,LSTM_KEY,GRU_KEY], type=str, default=LSTM_KEY)
parser.add_argument('-do','--drop_out',help='drop out', choices=[0.0,0.2,0.3],type=float,default=0.2)
parser.add_argument('-es','--embedding_size',help='embedding size', choices=[16, 32, 64, 256],type=int, default=128)
parser.add_argument('-bd','--bidirectional',help='bidirectional',choices=[True,False],type=bool,default=False)
parser.add_argument('-at','--attention',help='attention',choices=[True,False],type=bool,default=True)

args=parser.parse_args()


# Set the device type to CUDA if available, otherwise use CPU
is_gpu = torch.cuda.is_available()
if is_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

sweep_config = {
    "name" : "CS6910_Assignemnt3_With Attention",
    "method" : "random",
    'metric': {
        'name': VALIDATION_ACCURACY_TITLE,
        'goal': 'maximize'
    },
    "parameters" : {
        EMBEDDING_SIZE_KEY : {
          "values" : [16, 32, 64, 256]
        },
        EPOCHS_KEY : {
            "values" : [5,10]
        },
        ENCODER_LAYER_KEY: {
            "values": [1,2,3]
        },
        DECODER_LAYER_KEY: {
            "values": [1,2,3]
        },
        HIDDEN_LAYER_KEY:{
            "values": [16, 32, 64, 256]
        },
        IS_BIDIRECTIONAL_KEY:{
            "values": [True, False]
        },
        DROPOUT_KEY: {
            "values": [0,0.2,0.3]
        },
        CELL_TYPE_KEY: {
            "values": [RNN_KEY,GRU_KEY,LSTM_KEY]
        },
        LEARNING_RATE_KEY:{
            "values":[0.001,0.01]
        },
        BATCH_SIZE_KEY:{
            "values":[32,64,128]
        }
    }
}

"""# Utility Functions and classes"""

class Vocabulary:
    def __init__(self):
        self.str_count,self.int_encodding = dict(),dict()
        self.n_chars = 4
        self.str_encodding = {0: "<", 1: ">", 2: "?", 3: "."}

    def addWord(self, word):
        for char in word:
            try:
                self.str_count[char] += 1
            except:
                self.int_encodding[char] = self.n_chars
                self.str_encodding[self.n_chars] = char
                self.str_count[char] = 1
                self.n_chars += 1

def prepareData(dir):
    data = pd.read_csv(dir, sep=DELIMETER, names=[INPUT_LABEL, TARGET_LABEL])

    max_input_length = data[INPUT_LABEL].apply(len).max()
    max_target_length = data[TARGET_LABEL].apply(len).max()

    max_len=max(max_input_length,max_target_length)

    input_lang, output_lang = Vocabulary(), Vocabulary()

    pairs = pd.concat([data[INPUT_LABEL], data[TARGET_LABEL]], axis=1).values.tolist()

    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])

    return input_lang,output_lang,pairs,max_len



def prepareDataWithAttention(dir):

    data = pd.read_csv(dir, sep=DELIMETER, names=[INPUT_LABEL, TARGET_LABEL])

    max_input_length = data[INPUT_LABEL].apply(len).max()
    max_target_length = data[TARGET_LABEL].apply(len).max()


    input_lang, output_lang = Vocabulary(), Vocabulary()


    pairs = pd.concat([data[INPUT_LABEL], data[TARGET_LABEL]], axis=1).values.tolist()

    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])

    prepared_data = {
        "input_lang": input_lang,
        "output_lang": output_lang,
        "pairs": pairs,
        "max_input_length": max_input_length,
        "max_target_length": max_target_length,
    }

    return input_lang, output_lang, pairs, max_input_length, max_target_length

def writeToCSV(actual_X,actual_Y,predicted_Y,file_name):

    table_r = [[''.join(actual_X[i]),''.join(actual_Y[i]),''.join(predicted_Y[i])] for i in range(len(predicted_Y))]
    fields = [CSV_COLUMN_ACTUAL_X,CSV_COLUMN_ACTUAL_Y,CSV_COLUMN_PREDICETED_Y]

    # writing to csv file
    with open(file_name, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(table_r)



# helpTensorWithoutAttn
def helpTensor(lang, word, max_length):
    index_list = []
    for char in word:
        try:
            index_list.append(lang.char2index[char])
        except:
            index_list.append(SYMBOL_UNKNOWN)

    indexes = index_list
    indexes.append(SYMBOL_END)
    n = len(indexes)
    indexes.extend([SYMBOL_PADDING] * (max_length - n))
    result = torch.LongTensor(indexes)
    if is_gpu:
        return result.cuda()
    return result


def helpTensorWithAttention(lang, word, max_length):
    index_list=[]
    for char in word:
        try:
            index_list.append(lang.char2index[char])
        except:
            index_list.append(SYMBOL_UNKNOWN)
    indexes = index_list
    indexes.append(SYMBOL_END)
    n = len(indexes)
    indexes.extend([SYMBOL_PADDING] * (max_length - n))
    result = torch.LongTensor(indexes)
    if is_gpu:
        return result.cuda()
    return result

def makeTensor(input_lang, output_lang, pairs, reach):
    res = [(helpTensor(input_lang, pairs[i][0], reach), helpTensor(output_lang, pairs[i][1], reach)) for i in range(len(pairs))]
    return res

def accuracy(encoder, decoder, loader, batch_size, criterion, cell_type, num_layers_enc, max_length, output_lang, input_lang,is_test):
    with torch.no_grad():
        total = correct = 0
        actual_X = []
        actual_Y = []
        predicted_Y = []

        ignore = [SYMBOL_BEGIN, SYMBOL_END, SYMBOL_PADDING]
        for batch_x, batch_y in loader:
            encoder_hidden = encoder.initHidden(batch_size, num_layers_enc)

            input_variable = Variable(batch_x.transpose(0, 1))
            target_variable = Variable(batch_y.transpose(0, 1))

            if cell_type == LSTM_KEY:
                encoder_cell_state = encoder.initHidden(batch_size, num_layers_enc)
                encoder_hidden = (encoder_hidden, encoder_cell_state)


            output = torch.LongTensor(target_variable.size()[0], batch_size)

            for ei in range(input_variable.size()[0]):
                encoder_hidden = encoder(input_variable[ei], batch_size, encoder_hidden)[1]

            decoder_input = Variable(torch.LongTensor([SYMBOL_BEGIN] * batch_size))

            if is_test:
                x = None
                for i in range(batch_x.size()[0]):
                    x = [input_lang.str_encodding[letter.item()] for letter in batch_x[i] if letter not in ignore]
                    actual_X.append(x)

            if is_gpu:
                decoder_input = decoder_input.cuda()

            decoder_hidden = encoder_hidden

            # Decoder forward pass
            for di in range(target_variable.size()[0]):
                decoder_output, decoder_hidden = decoder(decoder_input, batch_size, decoder_hidden)
                topi = decoder_output.data.topk(1)[1]
                output[di], decoder_input = torch.cat(tuple(topi)), torch.cat(tuple(topi))
            output = output.transpose(0, 1)

            # Calculate accuracyWithoutAttn
            for di in range(output.size()[0]):
                sent = [output_lang.str_encodding[letter.item()] for letter in output[di] if letter not in ignore]
                y = [output_lang.str_encodding[letter.item()] for letter in batch_y[di] if letter not in ignore]
                if is_test:
                    actual_Y.append(y)
                    predicted_Y.append(sent)

                if sent == y:
                    correct += 1
                total += 1
        if is_test:
            writeToCSV(actual_X,actual_Y,predicted_Y)

    return (correct / total) * 100

def accuracyWithAttention(
    encoder,
    decoder,
    loader,
    batch_size,
    num_layers_enc,
    cell_type,
    output_lang,
    input_lang,
    criterion,
    max_length,
    is_test=False
    ):

    with torch.no_grad():

        total = correct = 0
        actual_X = []
        actual_Y = []
        predicted_Y = []

        for batch_x, batch_y in loader:

            encoder_hidden = encoder.initHidden(batch_size, num_layers_enc)

            input_variable = Variable(batch_x.transpose(0, 1))
            target_variable = Variable(batch_y.transpose(0, 1))

            if cell_type == LSTM_KEY:
                encoder_cell_state = encoder.initHidden(batch_size, num_layers_enc)
                encoder_hidden = (encoder_hidden, encoder_cell_state)


            output = torch.LongTensor(target_variable.size()[0], batch_size)

            encoder_outputs = Variable(
                torch.zeros(max_length, batch_size, encoder.hid_n)
            )

            if is_gpu:
                encoder_outputs = encoder_outputs.cuda()
            ignore = [SYMBOL_BEGIN, SYMBOL_END, SYMBOL_PADDING]

            if is_test:
                x = None
                for i in range(batch_x.size()[0]):
                    x = [input_lang.str_encodding[letter.item()] for letter in batch_x[i] if letter not in ignore]
                    actual_X.append(x)


            for ei in range(input_variable.size()[0]):
                encoder_output, encoder_hidden = encoder(
                    input_variable[ei], batch_size, encoder_hidden
                )
                encoder_outputs[ei] = encoder_output[0]

            decoder_input = Variable(torch.LongTensor([SYMBOL_BEGIN] * batch_size))
            if is_gpu:
                decoder_input = decoder_input.cuda()

            decoder_hidden = encoder_hidden

            for di in range(target_variable.size()[0]):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input,
                    batch_size,
                    decoder_hidden,
                    encoder_outputs.reshape(
                        batch_size, max_length, encoder.hid_n
                    ),
                )
                topi = decoder_output.data.topk(1)[1]
                decoder_input,output[di] = torch.cat(tuple(topi)),torch.cat(tuple(topi))
            output = output.transpose(0, 1)

            for di in range(output.size()[0]):
                sent = [output_lang.str_encodding[letter.item()] for letter in output[di] if letter not in ignore]
                y = [output_lang.str_encodding[letter.item()] for letter in batch_y[di] if letter not in ignore]

                if is_test:
                    actual_Y.append(y)
                    predicted_Y.append(sent)

                if sent == y:
                    correct += 1
                total += 1

        if is_test:
            writeToCSV(actual_X,actual_Y,predicted_Y,PREDICTION_WITH_ATTENSION_FILE_NAME)

    return (correct / total) * 100


# calc_lossWithoutAttn
def calc_loss(encoder, decoder, input_tensor, target_tensor, batch_size, encoder_optimizer, decoder_optimizer, criterion, cell_type, num_layers_enc, max_length, is_training, teacher_forcing_ratio=0.5):
    # Initialize the encoder hidden state
    output_hidden = encoder.initHidden(batch_size, num_layers_enc)

    # Check if LSTM and initialize cell state
    if cell_type == LSTM_KEY:
        encoder_cell_state = encoder.initHidden(batch_size, num_layers_enc)
        output_hidden = (output_hidden, encoder_cell_state)

    # Zero the gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Get input and target sequence lengths
    # input_length = input_tensor.size(0)
    # target_length = target_tensor.size(0)

    # Initialize loss
    loss = 0

    # Encoder forward pass
    for ei in range(input_tensor.size(0)):
        output_hidden = encoder(input_tensor[ei], batch_size, output_hidden)[1]

    # Initialize decoder input
    decoder_input = torch.LongTensor([SYMBOL_BEGIN] * batch_size)
    decoder_input = decoder_input.cuda() if is_gpu else decoder_input

    # Determine if using teacher forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Loop over target sequence
    if is_training:
        # Training phase
        for di in range(target_tensor.size(0)):
            decoder_output, output_hidden = decoder(decoder_input, batch_size, output_hidden)
            decoder_input = target_tensor[di] if use_teacher_forcing else decoder_output.argmax(dim=1)
            loss = criterion(decoder_output, target_tensor[di]) + loss
    else:
        # Validation phase
        with torch.no_grad():
            for di in range(target_tensor.size(0)):
                decoder_output, output_hidden = decoder(decoder_input, batch_size, output_hidden)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = decoder_output.argmax(dim=1)

    # Backpropagation and optimization in training phase
    if is_training:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    # Return the average loss per target length
    return loss.item() / target_tensor.size(0)

def calcLossWithAttention(
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    input_tensor,
    target_tensor,
    criterion,
    batch_size,
    cell_type,
    num_layers_enc,
    max_length,is_training,
    teacher_forcing_ratio=0.5,
):

    output_hidden = encoder.initHidden(batch_size, num_layers_enc)

    if cell_type == LSTM_KEY:
        encoder_cell_state = encoder.initHidden(batch_size, num_layers_enc)
        output_hidden = (output_hidden, encoder_cell_state)

    if is_training:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = Variable(torch.zeros(max_length, batch_size, encoder.hid_n))
    if is_gpu:
        encoder_outputs = encoder_outputs.cuda()
    loss = 0

    for ei in range(input_length):
        encoder_output, output_hidden = encoder(
            input_tensor[ei], batch_size, output_hidden
        )
        encoder_outputs[ei] = encoder_output[0]

    decoder_input = Variable(torch.LongTensor([SYMBOL_BEGIN] * batch_size))
    if is_gpu :
        decoder_input = decoder_input.cuda()
    decoder_hidden = output_hidden
    if is_training:
        use_teacher_forcing = False
        if random.random() < teacher_forcing_ratio:
            use_teacher_forcing = True
        n = target_length
        if not use_teacher_forcing :
            for di in range(n):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input,
                    batch_size,
                    decoder_hidden,
                    encoder_outputs.reshape(batch_size, max_length, encoder.hid_n),
                )
                #2 for loop ko bhar dal de
                topv, topi = decoder_output.data.topk(1)
                decoder_input = torch.cat(tuple(topi))
                if is_gpu :
                    decoder_input = decoder_input.cuda()
                loss += criterion(decoder_output, target_tensor[di])
        else:
            for di in range(n):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input,
                    batch_size,
                    decoder_hidden,
                    encoder_outputs.reshape(batch_size, max_length, encoder.hid_n),
                )
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]


        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
    else :
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input,
                batch_size,
                decoder_hidden,
                encoder_outputs.reshape(batch_size, max_length, encoder.hid_n),
            )
            topv, topi = decoder_output.data.topk(1)
            decoder_input = torch.cat(tuple(topi))

            if is_gpu:
                decoder_input = decoder_input.cuda()
            loss += criterion(decoder_output, target_tensor[di])

    avg_loss = loss.item() / target_length
    return avg_loss




# Train and evaluate the Seq2SeqWithoutAttn model
def seq2seq(encoder, decoder, train_loader, val_loader, test_loader, lr, optimizer, epochs, max_length_word, num_layers_enc, output_lang, input_lang, batch_size,cell_type,is_wandb):
    max_length = max_length_word - 1
    # Define the optimizer and criterion
    encoder_optimizer = optim.NAdam(encoder.parameters(), lr=lr) if optimizer == "nadam" else optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.NAdam(decoder.parameters(), lr=lr) if optimizer == "nadam" else optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        train_loss_total = 0
        val_loss_total = 0

        # Training phase
        for batch_x, batch_y in train_loader:
            batch_x = Variable(batch_x.transpose(0, 1))
            batch_y = Variable(batch_y.transpose(0, 1))
            # Calculate the training loss
            loss = calc_loss(encoder, decoder, batch_x, batch_y, batch_size, encoder_optimizer, decoder_optimizer, criterion, cell_type, num_layers_enc, max_length, is_training=True)
            train_loss_total += loss

        train_loss_avg = train_loss_total / len(train_loader)
        print(f"Epoch: {epoch} | Train Loss: {train_loss_avg:.4f} |", end="")

        # Validation phase
        for batch_x, batch_y in val_loader:
            batch_x = Variable(batch_x.transpose(0, 1))
            batch_y = Variable(batch_y.transpose(0, 1))
            # Calculate the validation loss
            loss = calc_loss(encoder, decoder, batch_x, batch_y, batch_size, encoder_optimizer, decoder_optimizer, criterion, cell_type, num_layers_enc, max_length, is_training=False)
            val_loss_total += loss

        val_loss_avg = val_loss_total / len(val_loader)
        print(f"Val Loss: {val_loss_avg:.4f} |", end="")

        train_acc = accuracy(
            encoder= encoder,
            decoder = decoder,
            loader = train_loader,
            batch_size = batch_size,
            criterion = criterion,
            cell_type = cell_type,
            num_layers_enc = num_layers_enc,
            max_length = max_length,
            output_lang = output_lang,
            input_lang = input_lang,
            is_test = True
            )
        train_acc /= 100
        print(f"train Accuracy: {train_acc:.4%} |", end="")

        # Calculate validation accuracyWithoutAttn
        val_acc = accuracy(
            encoder= encoder,
            decoder = decoder,
            loader = val_loader,
            batch_size = batch_size,
            criterion = criterion,
            cell_type = cell_type,
            num_layers_enc = num_layers_enc,
            max_length = max_length,
            output_lang = output_lang,
            input_lang = input_lang,
            is_test = True
            )

        val_acc /= 100
        print(f"Val Accuracy: {val_acc:.4%} |", end="")

        test_acc = accuracy(
            encoder= encoder,
            decoder = decoder,
            loader = test_loader,
            batch_size = batch_size,
            criterion = criterion,
            cell_type = cell_type,
            num_layers_enc = num_layers_enc,
            max_length = max_length,
            output_lang = output_lang,
            input_lang = input_lang,
            is_test = True
            )
        test_acc = accuracy(encoder, decoder, test_loader, batch_size, criterion, cell_type, num_layers_enc, max_length, output_lang, input_lang)
        test_acc /= 100
        print(f"Test Accuracy: {test_acc:.4%}")
        if is_wandb:
            wandb.log(
                {
                    TRAIN_ACCURACY_TITLE: train_acc,
                    VALIDATION_ACCURACY_TITLE: val_acc,
                    TEST_ACCURACY_TITLE: test_acc,
                    TRAIN_LOSS_TITLE: train_loss_avg,
                    VALIDATION_LOSS_TITLE: val_loss_avg,
                    # TEST_LOSS_TITLE: test_loss
                }
            )


def seq2seqWithAttention(
    encoder,
    decoder,
    train_loader,
    val_loader,
    test_loader,
    learning_rate,
    optimizer,
    epochs,
    max_length_word,
    attention,
    num_layers_enc,
    output_lang,
    input_lang,
    batch_size,
    cell_type,
    is_wandb
 ):
    max_length = max_length_word - 1
    n_val = len(val_loader)
    n_train = len(train_loader)
    encoder_optimizer = (
        optim.NAdam(encoder.parameters(), lr=learning_rate)
        if optimizer == "nadam"
        else optim.Adam(encoder.parameters(), lr=learning_rate)
    )
    decoder_optimizer = (
        optim.NAdam(decoder.parameters(), lr=learning_rate)
        if optimizer == "nadam"
        else optim.Adam(decoder.parameters(), lr=learning_rate)
    )
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        train_loss_total, val_loss_total = 0, 0

        for batchx, batchy in train_loader:
            batchx = Variable(batchx.transpose(0, 1))
            batchy = Variable(batchy.transpose(0, 1))
            loss = calcLossWithAttention(
                encoder = encoder,
                decoder =decoder,
                encoder_optimizer = encoder_optimizer,
                decoder_optimizer = decoder_optimizer,
                input_tensor = batchx,
                target_tensor = batchy,
                criterion = criterion,
                batch_size =  batch_size,
                cell_type = cell_type,
                num_layers_enc = num_layers_enc,
                max_length = max_length + 1,
                is_training = True, #is_training
                teacher_forcing_ratio=0.5
            )
            train_loss_total = train_loss_total + loss
        train_loss_avg = train_loss_total / n_train
        print(f"Epoch: {epoch} | Train Loss: {train_loss_avg:.4f} | ", end="")

        for batchx, batchy in val_loader:
            batchx = Variable(batchx.transpose(0, 1))
            batchy = Variable(batchy.transpose(0, 1))
            loss = calcLossWithAttention(
                encoder = encoder,
                decoder = decoder,
                encoder_optimizer = encoder_optimizer,
                decoder_optimizer = decoder_optimizer,
                input_tensor = batchx,
                target_tensor = batchy,
                criterion = criterion,
                batch_size =  batch_size,
                cell_type = cell_type,
                num_layers_enc = num_layers_enc,
                max_length = max_length + 1,
                is_training = False, #is_training
                teacher_forcing_ratio=0.5
            )

            val_loss_total = val_loss_total+ loss

        val_loss_avg = val_loss_total / n_val
        print(f"Val Loss: {val_loss_avg:.4f} | ", end="")

        train_acc = accuracyWithAttention(
            encoder = encoder,
            decoder = decoder,
            loader = train_loader,
            batch_size = batch_size,
            num_layers_enc = num_layers_enc,
            cell_type = cell_type,
            output_lang = output_lang,
            input_lang = input_lang,
            criterion = criterion,
            max_length = max_length + 1,
        )
        train_acc = train_acc /  100
        print(f"Train Accuracy: {train_acc:.4%} |", end="")

        val_acc = accuracyWithAttention(
            encoder = encoder,
            decoder = decoder,
            loader = val_loader,
            batch_size = batch_size,
            num_layers_enc = num_layers_enc,
            cell_type = cell_type,
            output_lang = output_lang,
            input_lang = input_lang,
            criterion = criterion,
            max_length = max_length + 1,
        )
        val_acc = val_acc / 100
        print(f"Val Accuracy: {val_acc:.4%} |", end="")
        test_acc = accuracyWithAttention(
            encoder = encoder,
            decoder = decoder,
            loader = test_loader,
            batch_size = batch_size,
            num_layers_enc = num_layers_enc,
            cell_type = cell_type,
            output_lang = output_lang,
            input_lang = input_lang,
            criterion = criterion,
            max_length = max_length + 1,
        )

        test_acc = test_acc / 100
        print(f"Test Accuracy: {test_acc:.4%}")

        if is_wandb:
            wandb.log(
                {
                    TRAIN_ACCURACY_TITLE: train_acc,
                    VALIDATION_ACCURACY_TITLE: val_acc,
                    TEST_ACCURACY_TITLE: test_acc,
                    TRAIN_LOSS_TITLE: train_loss_avg,
                    VALIDATION_LOSS_TITLE: val_loss_avg,
                    # TEST_LOSS_TITLE: test_loss
                }
            )

def store_heatmaps(encoder, decoder, loader, give_batch_size, given_num_layers_encoder,cell_type, input_lang,max_length,bi_directional,output_lang):

    temp = give_batch_size
    # Evaluating for 10 test inputs so batch size is set to 1
    give_batch_size = 1

    # disabled gradient calculation for inference, helps reduce memory consumption for computations
    with torch.no_grad():

        # Need heatmaps for 10 inputs only that will be ensured by count
        count = 0
        # Need to store predicted y's, x's and attentions corresponding to these 10 inputs
        predictions,xs,attentions = [],[],[]

        # Loops until 10 points have been covered
        for batch_x, batch_y in loader:
            count+=1
            # Fetch batch size and number of layers from configuration dictionary
            batch_size,num_layers_enc = give_batch_size,given_num_layers_encoder
            # Initial encoder hidden is set to a tensor of zeros using initHidden function
            encoder_hidden = encoder.initHidden(batch_size,num_layers_enc)

            decoder_attentions = torch.zeros(max_length, batch_size, max_length)

            # Transpose the batch_x and batch_y tensors
            input_variable, target_variable = batch_x.transpose(0, 1), batch_y.transpose(0, 1)

            # Incase of LSTM cell the encoder hidden is a tupple of (encoder hidden, encoder cell state) while for the other two it is just encoder hidden
            if cell_type == LSTM_KEY:
                encoder_hidden = (encoder_hidden, encoder.initHidden(batch_size,num_layers_enc))

            # Finding the length of input and target tensors

            input_var_n,target_var_n = input_variable.size(),target_variable.size()

            ip_n,op_n = input_var_n[0],target_var_n[0]

            # Appending the input word
            x = None
            for i in range(batch_x.size()[0]):
                x = [input_lang.str_encodding[letter.item()] for letter in batch_x[i] if letter not in [SYMBOL_BEGIN, SYMBOL_END, SYMBOL_PADDING, SYMBOL_UNKNOWN]]
                xs.append(x)

            # Initializing output as a tensor of size target_length X batch_size
            output = torch.LongTensor(op_n, batch_size)

            # In attention mechanism we need encoder ouputs at every time step to be stored and so we will initialize a tensor that will be used to store them
            encoder_outputs = None
            temp_hid_size = encoder.hid_n
            if bi_directional :
                # Incase of bidirectional hidden size needs to be doubled
                temp_hid_size = encoder.hid_n*2
            encoder_outputs = Variable(torch.zeros(max_length, batch_size, temp_hid_size))
                # Shift encoder outputs to cuda if available
            if is_gpu:
                encoder_outputs = encoder_outputs.cuda()
            # Passing ith character of every word from the input batch into the encoder iteratively
            for i in range(ip_n):
                encoder_result = encoder(input_variable[i], batch_size, encoder_hidden)

                encoder_output = encoder_result[0]
                encoder_hidden = encoder_result[1]

                encoder_outputs[i] = encoder_output[0]

            # Setting a list of start of word tokens to pass into the decoder initially and converting it into tensors
            sow_list = [SYMBOL_BEGIN] * batch_size
            decoder_input = torch.LongTensor(sow_list)
            # Shift decoder_inputs to cuda if available
            if is_gpu :
                decoder_input = decoder_input.cuda()

            # decoder_hidden is set to the final encoder_hidden after the encoder loop completes
            decoder_hidden = encoder_hidden
            # Initialize them before using them
            decoder_output,decoder_attention = None,None

            # We are just evaluating the output of decoder in this case so we don't use teacher forcing here
            # We give the decoder input to be the best predictions for i+1th charcter for the whole batch from ith time step
            for i in range(op_n):
                enc_hid_size =  encoder.hid_n
                if bi_directional :
                    enc_hid_size = enc_hid_size*2

                decoder_output, decoder_hidden, decoder_attention= decoder(decoder_input, batch_size, decoder_hidden, encoder_outputs.reshape(batch_size,max_length, encoder.hid_n))
                # Best prediction comes from using topk(k=1) function

                decoder_attentions[i] = decoder_attention.data
                temp = decoder_output.data
                topi_result = temp.topk(1)
                decoder_input = topi_result[1]
                output[i] = torch.cat(tuple(topi_result[1]))

            # Appending the attentions for every input word
            attentions.append(decoder_attentions)

            # Taking transpose of output and finding it's length
            output = output.transpose(0,1)
            op_len = output.size()
            sent = list()
            for i in range(op_len[0]):
                # sent is ith predicted word of a batch
                for letter in output[i]:
                    if letter not in [SYMBOL_BEGIN, SYMBOL_END, SYMBOL_PADDING, SYMBOL_UNKNOWN]:
                        sent.append(output_lang.str_encodding[letter.item()])
                # Appending the predicted words for the input word
                predictions.append(sent)

            if count == 12 :
                give_batch_size = temp
                # Returns input words, predicted words, attentions respectively
                return predictions,attentions,xs

def plot_heatmap(test_loader, encoder, decoder,batch_size,num_layers_encoder,cell_type,max_length_word,input_lang,output_lang,bi_directional):
    max_length = max_length_word
    # input words, predicted words, attentions respectively fetched from store_heatmaps
    predictions,atte,test_english = store_heatmaps(
        encoder = encoder,
        decoder=decoder,
        loader=test_loader,
        give_batch_size=batch_size,
        given_num_layers_encoder=num_layers_encoder,
        cell_type=cell_type,
        input_lang=input_lang,
        max_length=max_length+1,
        bi_directional=bi_directional,
        output_lang = output_lang
        )
    # fig will store the figure with 10 subplots
    heat_map_plot = []
    n = 12
    heat_map_plot , axs = plt.subplots(4,3)
    heat_map_plot.set_size_inches(23, 15)
    l,k = -1,0
    # Iterate 12 times
    while i<n:
        attn_weights = []
        # Fetch attention corresponding to ith input word
        attn_weight = atte[i].reshape(-1,max_length+1)
        ylabel,xlabel = [""],[""]
        # ylabel will have predicted word
        ylabel += [char for char in predictions[i]]
        # xlabel will have input word
        xlabel += [char for char in test_english[i]]

        # y will be of size of ylable
        pred_n = len(predictions[i])+1
        for j in range(1,pred_n):
            # x will be of size of xlabel
            temp = len(xlabel)
            fg = attn_weight[j][1:temp]
            fg_arr = fg.numpy()
            attn_weights.append(fg_arr)

        attn_weights = attn_weights[:-1]
        # After every 3 goto next line
        if i%3 == 0:
            l = l + 1
            k = 0

        axs[l][k].set_xticklabels(xlabel)
        # set ylabels with support for hindi text
        xyz = FontProperties(fname = WANDB_FONT_FAMILY, size = 10)
        axs[l][k].set_yticklabels(ylabel, fontproperties = xyz)
        k+=1
        i+=1
    # Plot on wandb
    run = wandb.init(project=WANDB_PROJECT_NAME,entity = WANDB_ENTITY_NAME)
    plt.show()
    wandb.log({'heatmaps':heat_map_plot})
    wandb.finish()

def beam_search(prepared_d, bw, lp, cell_type, en_de_model, letter):
    TARGET_REV_KEY = "output_index_rev"
    input_x_dim = prepared_d[MAX_LEN_KEY]+1
    result = np.zeros((input_x_dim, 1), dtype=np.int32)
    for idx, char in enumerate(letter):
        temp = prepared_d[INPUT_INDEX_KEY]
        result[idx, 0] = temp[char]
    temp = prepared_d[INPUT_INDEX_KEY]
    result[idx + 1, 0] = temp[prepared_d[TARGET_LANG_KEY]]
    result = torch.tensor(result, dtype=torch.int32).to(device)
    with torch.no_grad():
        if cell_type == LSTM_KEY:
            hd, cell = en_de_model.encoder(result)
        else:
            hd = en_de_model.encoder(result)
    output_start = prepared_d[OUTPUT_INDEX_KEY][prepared_d[SOURCE_LANG_KEY]]
    out_reshape, hidden_par  = np.array(output_start).reshape(1,), hd.unsqueeze(0)
    initial_sequence = torch.tensor(out_reshape).to(device)
    beam = []
    beam.append((0.0, initial_sequence, hidden_par))
    n = len(prepared_d[OUTPUT_INDEX_KEY])
    for _ in range(n):
        candidates = []
        for b in beam:
            score = b[0]
            seq = b[1]
            hd = b[2]
            temp_prepare_d = prepared_d[OUTPUT_INDEX_KEY]
            if seq[-1].item() == temp_prepare_d[prepared_d[TARGET_LANG_KEY]]:
                temp_candidate = (score, seq, hd)
                candidates.append(temp_candidate)
                continue
            temp_layer_int = np.array(seq[-1].item()).reshape(1, )
            temp_layer = [temp_layer_int, hd.squeeze(0)]
            reshape_last = temp_layer[0]
            hdn = temp_layer[1]
            x = torch.tensor(reshape_last).to(device)
            if cell_type == LSTM_KEY:
                op, hd, cell = en_de_model.decoder(x, hdn, cell)
            else:
                op, hd = en_de_model.decoder(x, hdn, None)
            top_k_result_item = torch.topk(Function.softmax(op, dim=1), k = bw)
            cond_topk = top_k_result_item[0]
            label_topk = top_k_result_item[1]
            for topk_item in zip(cond_topk[0], label_topk[0]):
                approx = topk_item[0]
                label = topk_item[1]
                unsquzeed_label = label.unsqueeze(0)
                new_seq = torch.cat((seq, unsquzeed_label), dim=0)
                temp_log_approx = torch.log(approx)
                temp_log = temp_log_approx.item()
                partial_exp = ((len(new_seq) - 1) / 5)
                deno = ( partial_exp ** lp)
                unsquezed_hidden_layer = hd.unsqueeze(0)
                candidates.append((score + temp_log / deno, new_seq,unsquezed_hidden_layer ))
        beam = heapq.nlargest(bw, candidates, key=lambda x: x[0])
    beamseq_result = max(beam, key=lambda x: x[0])
    final_resut = []
    for token in beamseq_result[1][1:]:
        final_resut.append(prepared_d[TARGET_REV_KEY][token.item()])
    return ''.join(final_resut)[:-1]

"""# Encoder Class"""

# EncoderRNNWithoutAttn
class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers_encoder, cell_type, drop_out, bi_directional):
        super(EncoderRNN, self).__init__()

        self.emb_n = embedding_size
        self.hid_n = hidden_size
        self.encoder_n = num_layers_encoder
        self.model_key = cell_type
        self.is_dropout = drop_out
        self.is_bi_dir = bi_directional

        self.embedding = nn.Embedding(input_size, self.emb_n)
        self.dropout = nn.Dropout(self.is_dropout)

        cell_map = dict({RNN_KEY: nn.RNN, GRU_KEY: nn.GRU, LSTM_KEY: nn.LSTM})
        self.cell_layer = cell_map[self.model_key](
            input_size = self.emb_n,
            hidden_size = self.hid_n,
            num_layers=self.encoder_n,
            dropout=self.is_dropout,
            bidirectional=self.is_bi_dir,
        )

    def forward(self, input, batch_size, hidden):
        pre_embedded = self.embedding(input)
        transformed_embedded_data = pre_embedded.view(1, batch_size, -1)
        embedded = self.dropout(transformed_embedded_data)
        y_cap, hidden = self.cell_layer(embedded, hidden)
        return y_cap, hidden

    def initHidden(self, batch_size, num_layers_enc):
        if self.is_bi_dir:
            weights = torch.zeros(num_layers_enc * 2 , batch_size, self.hid_n)
        else:
            weights = torch.zeros(num_layers_enc, batch_size, self.hid_n)

        if is_gpu:
            return weights.cuda()
        return weights


class EncoderRNNWithAttention(nn.Module):
    def __init__(self, input_size, embedding_size,hidden_size,num_layers_encoder,cell_type,drop_out,bi_directional):
        super(EncoderRNNWithAttention, self).__init__()

        self.emb_n = embedding_size
        self.hid_n = hidden_size
        self.encoder_n = num_layers_encoder
        self.model_key = cell_type
        self.is_dropout = drop_out
        self.is_bi_dir = bi_directional

        self.embedding = nn.Embedding(input_size, self.emb_n)
        self.dropout = nn.Dropout(self.is_dropout)

        cell_map = dict({RNN_KEY: nn.RNN, GRU_KEY: nn.GRU, LSTM_KEY: nn.LSTM})
        self.cell_layer = cell_map[self.model_key](
            input_size = self.emb_n,
            hidden_size = self.hid_n,
            num_layers=self.encoder_n,
            dropout=self.is_dropout,
            bidirectional=self.is_bi_dir,
        )

    def forward(self, input, batch_size, hidden):
        pre_embedded = self.embedding(input)
        transformed_embedded_data = pre_embedded.view(1, batch_size, -1)
        embedded = self.dropout(transformed_embedded_data)
        y_cap, hidden = self.cell_layer(embedded, hidden)
        return y_cap, hidden

    def initHidden(self, batch_size, num_layers_enc):
        if self.is_bi_dir:
            weights = torch.zeros(num_layers_enc * 2 , batch_size, self.hid_n)
        else:
            weights = torch.zeros(num_layers_enc, batch_size, self.hid_n)

        if is_gpu:
            return weights.cuda()
        return weights

"""# Decoder Class"""

# DecoderRNNWithoutAttn
class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers_decoder, cell_type, drop_out, bi_directional, output_size):
        super(DecoderRNN, self).__init__()

        self.emb_n = embedding_size
        self.hid_n = hidden_size
        self.decoder_n = num_layers_decoder
        self.model_key = cell_type
        self.is_dropout = drop_out
        self.is_bi_dir = bi_directional

        # Create an embedding layer
        self.embedding = nn.Embedding(output_size, self.emb_n)
        self.dropout = nn.Dropout(self.is_dropout)

        cell_map = dict({RNN_KEY: nn.RNN, GRU_KEY: nn.GRU, LSTM_KEY: nn.LSTM})

        if self.cell_type in cell_map:
            self.cell_layer = cell_map[self.model_key](
                input_size = self.emb_n,
                hidden_size = self.hid_n,
                num_layers=self.decoder_n,
                dropout=self.is_dropout,
                bidirectional=self.is_bi_dir,
            )

        # Linear layer for output
        if self.is_bi_dir :
            self.out = nn.Linear(self.hid_n * 2, output_size)
        else:
            self.out = nn.Linear(self.hid_n,output_size)

        # Softmax activation
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, batch_size, hidden):
        y_cap = Function.relu(self.dropout(self.embedding(input).view(1, batch_size, -1)))
        y_cap, hidden = self.cell_layer(y_cap, hidden)

        y_cap = self.softmax(self.out(y_cap[0]))
        return y_cap, hidden


class DecoderRNNWithAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        embedding_size,
        cell_type,
        num_layers_decoder,
        drop_out,
        max_length_word,
        output_size,
    ):

        super(DecoderRNNWithAttention, self).__init__()

        self.hid_n = hidden_size
        self.emb_n = embedding_size
        self.model_key = cell_type
        self.decoder_n = num_layers_decoder
        self.drop_out = drop_out
        self.max_length_word = max_length_word

        self.embedding = nn.Embedding(output_size, embedding_dim=self.emb_n)
        layer_size = self.emb_n + self.hid_n
        self.attention_layer = nn.Linear(
            layer_size , self.max_length_word
        )
        self.attention_combine = nn.Linear(
            layer_size, self.emb_n
        )
        self.dropout = nn.Dropout(self.drop_out)

        self.cell_layer = None
        cell_map = dict({RNN_KEY: nn.RNN, GRU_KEY: nn.GRU, LSTM_KEY: nn.LSTM})

        if self.model_key in cell_map:
            self.cell_layer = cell_map[self.model_key](
                self.emb_n,
                self.hid_n,
                num_layers=self.decoder_n,
                dropout=self.drop_out,
            )

        self.out = nn.Linear(self.hid_n, output_size)

    def forward(self, input, batch_size, hidden, encoder_outputs):
        pre_embedded = self.embedding(input)
        embedded = pre_embedded.view(1, batch_size, -1)

        attention_weights = None

        attention_weights = Function.softmax(
            self.attention_layer(torch.cat((embedded[0],hidden[0][0] if self.model_key == LSTM_KEY else hidden[0]), 1)), dim=1
        )

        attention_applied = torch.bmm(
            attention_weights.view(batch_size, 1, self.max_length_word),
            encoder_outputs,
        ).view(1, batch_size, -1)

        y_cap = torch.cat((embedded[0], attention_applied[0]), 1)
        y_cap = Function.relu(self.attention_combine(y_cap).unsqueeze(0))
        # if self.cell_type=RNN" :
        y_cap, hidden = self.cell_layer(y_cap, hidden)
        y_cap = Function.log_softmax(self.out(y_cap[0]), dim=1)

        return y_cap, hidden, attention_weights

"""# Train Function"""

def train(config_defaults = best_params,flag = False,is_wandb = False,is_heat_map = False):
    optimizer = NADAM_KEY
    if is_wandb:
        wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,config = config_defaults)
        args = wandb.config
        # Set the name of the run

        wandb.run.name = 'ep-'+str(args[EPOCHS_KEY])+'-lr-'+str(args[LEARNING_RATE_KEY])+'-bs-'+str(args[BATCH_SIZE_KEY])+'-el-'+str(args[ENCODER_LAYER_KEY])+'-dl-'+str(args[DECODER_LAYER_KEY]) \
                        +'-hl-'+str(args[HIDDEN_LAYER_KEY])+'-do-'+ str(args[DROPOUT_KEY])+ '-es-'+str(args[EMBEDDING_SIZE_KEY]) \
                        + '-is_bd-'+str(args[IS_BIDIRECTIONAL_KEY])+'-model'+str(args[CELL_TYPE_KEY])

    if flag:

        input_langs, output_langs, pairs, max_input_length, max_target_length = prepareDataWithAttention(TRAIN_DATASET_PATH)
        print("train:sample:", random.choice(pairs))
        print(f"Number of training examples: {len(pairs)}")

        # validation
        _,_,val_pairs,_,_ = prepareDataWithAttention(VALIDATION_DATASET_PATH)

        print("validation:sample:", random.choice(val_pairs))
        print(f"Number of validation examples: {len(val_pairs)}")
        # Test
        _,_,test_pairs,_,_ = prepareDataWithAttention(TEST_DATASET_PATH)
        print("Test:sample:", random.choice(test_pairs))
        print(f"Number of Test examples: {len(test_pairs)}")

        max_len = max(max_input_length, max_target_length) + 3
        print(max_len)

        pairs = makeTensor(input_lang=input_langs,output_lang= output_langs,pairs= pairs, reach=max_len)
        val_pairs = makeTensor(input_lang=input_langs,output_lang= output_langs,pairs= val_pairs, reach=max_len)
        test_pairs = makeTensor(input_lang=input_langs,output_lang= output_langs,pairs= test_pairs, reach=max_len)

        train_loader = DataLoader(pairs, batch_size=config_defaults[BATCH_SIZE_KEY], shuffle=True)
        val_loader = DataLoader(val_pairs, batch_size=config_defaults[BATCH_SIZE_KEY], shuffle=True)
        test_loader = DataLoader(test_pairs, batch_size=config_defaults[BATCH_SIZE_KEY], shuffle=True)

        encoder1 = EncoderRNNWithAttention(
            input_size = input_langs.n_chars,
            embedding_size =  config_defaults[EMBEDDING_SIZE_KEY],
            hidden_size =  config_defaults[HIDDEN_LAYER_KEY],
            num_layers_encoder = config_defaults[ENCODER_LAYER_KEY],
            cell_type = config_defaults[CELL_TYPE_KEY],
            drop_out = config_defaults[DROPOUT_KEY],
            bi_directional = config_defaults[IS_BIDIRECTIONAL_KEY]
            )

        attndecoder1 = DecoderRNNWithAttention(
            embedding_size = config_defaults[EMBEDDING_SIZE_KEY],
            hidden_size = config_defaults[HIDDEN_LAYER_KEY],
            num_layers_decoder = config_defaults[ENCODER_LAYER_KEY],
            cell_type = config_defaults[CELL_TYPE_KEY],
            drop_out = config_defaults[DROPOUT_KEY],
            # bi_directional = config_defaults[IS_BIDIRECTIONAL_KEY],
            max_length_word = max_len,
            output_size = output_langs.n_chars
            )

        if is_gpu== True:
            encoder1 = encoder1.cuda()
            attndecoder1 = attndecoder1.cuda()
        print("with attention")
        attention = True
        seq2seqWithAttention(
            encoder = encoder1,
            decoder = attndecoder1,
            train_loader = train_loader,
            val_loader = val_loader,
            test_loader = test_loader,
            learning_rate = config_defaults[LEARNING_RATE_KEY],
            optimizer = optimizer,
            epochs = config_defaults[EPOCHS_KEY],
            max_length_word = max_len,
            attention=attention,
            num_layers_enc = config_defaults[ENCODER_LAYER_KEY],
            output_lang = output_langs,
            input_lang = input_langs,
            batch_size = config_defaults[BATCH_SIZE_KEY],
            cell_type = config_defaults[CELL_TYPE_KEY],
            is_wandb = is_wandb
        )
        if is_heat_map:
            plot_heatmap(
                test_loader = test_loader,
                encoder = encoder1,
                decoder = attndecoder1,
                batch_size = config_defaults[BATCH_SIZE_KEY],
                num_layers_encoder = config_defaults[ENCODER_LAYER_KEY],
                cell_type = config_defaults[CELL_TYPE_KEY],
                max_length_word = max_len,
                input_lang = input_langs,
                output_lang = output_langs,
                bi_directional = config_defaults[IS_BIDIRECTIONAL_KEY]
                )

    else:
        # Prepare training data
        input_langs,output_langs,pairs,max_len = prepareData(TRAIN_DATASET_PATH)
        print("train:sample:", random.choice(pairs))
        train_n = len(pairs)
        print(f"Number of training examples: {train_n}")

        # Prepare validation data
        input_langs,output_langs,val_pairs,max_len_val = prepareData(VALIDATION_DATASET_PATH)
        val_n = len(val_pairs)
        print("validation:sample:", random.choice(val_pairs))
        print(f"Number of validation examples: {val_n}")

        # Prepare test data
        input_langs,output_langs,test_pairs,max_len_test = prepareData(TEST_DATASET_PATH)
        test_n = len(test_pairs)
        print("Test:sample:", random.choice(test_pairs))
        print(f"Number of Test examples: {test_n}")

        max_len = max(max_len, max(max_len_val, max_len_test)) + 4
        print(max_len)

        # Convert data to tensors and create data loaders
        pairs = makeTensor(input_langs, output_langs, pairs, max_len)
        val_pairs = makeTensor(input_langs, output_langs, val_pairs, max_len)
        test_pairs = makeTensor(input_langs, output_langs, test_pairs, max_len)

        train_loader = DataLoader(dataset = pairs, batch_size=config_defaults[BATCH_SIZE_KEY], shuffle=True)
        val_loader = DataLoader(dataset = val_pairs, batch_size=config_defaults[BATCH_SIZE_KEY], shuffle=True)
        test_loader = DataLoader(dataset = test_pairs, batch_size=config_defaults[BATCH_SIZE_KEY], shuffle=True)

        # Create the encoder and decoder models
        encoder1 = EncoderRNN(
            input_size = input_langs.n_chars,
            embedding_size =  config_defaults[EMBEDDING_SIZE_KEY],
            hidden_size =  config_defaults[HIDDEN_LAYER_KEY],
            num_layers_encoder = config_defaults[ENCODER_LAYER_KEY],
            cell_type = config_defaults[CELL_TYPE_KEY],
            drop_out = config_defaults[DROPOUT_KEY],
            bi_directional = config_defaults[IS_BIDIRECTIONAL_KEY]
            )
        decoder1 = DecoderRNN(
            embedding_size = config_defaults[EMBEDDING_SIZE_KEY],
            hidden_size = config_defaults[HIDDEN_LAYER_KEY],
            num_layers_decoder = config_defaults[ENCODER_LAYER_KEY],
            cell_type = config_defaults[CELL_TYPE_KEY],
            drop_out = config_defaults[DROPOUT_KEY],
            bi_directional = config_defaults[IS_BIDIRECTIONAL_KEY],
            output_size = output_langs.n_chars
            )

        if is_gpu:
            encoder1, decoder1 = encoder1.cuda(), decoder1.cuda()

        print("vanilla seq2seqWithoutAttn")
        # Train and evaluate the Seq2SeqWithoutAttn model
        seq2seq(
            encoder = encoder1,
            decoder = decoder1,
            train_loader = train_loader,
            val_loader = val_loader,
            test_loader = test_loader,
            lr = config_defaults[LEARNING_RATE_KEY],
            optimizer = optimizer,
            epochs = config_defaults[EPOCHS_KEY],
            max_length_word = max_len,
            num_layers_enc = config_defaults[ENCODER_LAYER_KEY],
            output_lang = output_langs,
            input_langs = input_langs,
            batch_size = config_defaults[BATCH_SIZE_KEY],
            cell_type = config_defaults[CELL_TYPE_KEY],
            is_wandb = is_wandb
            )
        if is_heat_map:
            plot_heatmap(
                test_loader = test_loader,
                encoder = encoder1,
                decoder = decoder1,
                batch_size = config_defaults[BATCH_SIZE_KEY],
                num_layers_encoder = config_defaults[ENCODER_LAYER_KEY],
                cell_type = config_defaults[CELL_TYPE_KEY],
                max_length_word = max_len,
                input_lang = input_langs,
                output_lang = output_langs,
                bi_directional = config_defaults[IS_BIDIRECTIONAL_KEY]
                )


def run_sweep():
    sweep_id = wandb.sweep(sweep_config, project="dl-assignment-3", entity="cs23m007")
    print('sweep_id: ', sweep_id)
    train(best_params,flag=True,is_wandb= True,is_heat_map= False)
    wandb.finish()