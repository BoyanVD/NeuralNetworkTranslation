#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import torch
import random

class Encoder(torch.nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout=0.5):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self. num_layers = num_layers

        self.dropout = torch.nn.Dropout(dropout)
        self.embedding = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.linear_hidden = torch.nn.Linear(hidden_size*2, hidden_size)
        self.linear_cell = torch.nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        encoder_state_vectors, (hidden, cell) = self.rnn(embedded)

        forward_hidden = hidden[0:1]
        backward_hidden = hidden[1:2]
        hidden = self.linear_hidden(torch.cat((forward_hidden, backward_hidden), dim=2))

        forward_cell = cell[0:1]
        backward_cell = cell[1:2]
        cell = self.linear_hidden(torch.cat((forward_cell, backward_cell), dim=2))

        return encoder_state_vectors, hidden, cell

class Decoder(torch.nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout): # input_size = output_size
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = torch.nn.Dropout(dropout)
        self.embedding = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.LSTM(hidden_size*2 + embedding_size, hidden_size, num_layers)
        self.energy = torch.nn.Linear(hidden_size*3, 1)
        self.softmax = torch.nn.Softmax(dim=0)
        self.relu = torch.nn.ReLU()
        self.proj = torch.nn.Linear(hidden_size, output_size)


    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))

        sequence_length = encoder_states.shape[0]
        decoder_hidden_reshaped = hidden.repeat(sequence_length, 1, 1)

        energy = self.relu(self.energy(torch.cat((decoder_hidden_reshaped, encoder_states), dim=2)))
        attention = self.softmax(energy)
        attention = attention.permute(1, 2, 0)
        encoder_states = encoder_states.permute(1, 0, 2)
        context = torch.bmm(attention, encoder_states).permute(1, 0, 2)

        rnn_input = torch.cat((context, embedding), dim=2)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        predictions = self.proj(outputs)
        prediction = predictions.squeeze(0)

        return prediction, hidden, cell

class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))

    def save(self,fileName):
        torch.save(self.state_dict(), fileName)

    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))

    def __init__(self, encoder, decoder, sourceWord2ind, targetWord2ind, unkToken, padToken, endToken, startToken):
        super(NMTmodel, self).__init__()

        self.sourceWord2ind = sourceWord2ind
        self.targetWord2ind = targetWord2ind

        self.unkTokenIdx = sourceWord2ind[unkToken]
        self.padTokenIdx = sourceWord2ind[padToken]
        self.endTokenIdx = sourceWord2ind[endToken]
        self.startTokenIdx = sourceWord2ind[startToken]

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        src = self.preparePaddedBatch(source, self.sourceWord2ind)
        trg = self.preparePaddedBatch(target, self.targetWord2ind)

        batch_size = src.shape[1]
        target_length = trg.shape[0]
        target_vocabulary_size = len(self.targetWord2ind)
        device = next(self.parameters()).device

        outputs = torch.zeros(target_length, batch_size, target_vocabulary_size).to(device)

        encoder_states, hidden, cell = self.encoder(src)

        x = trg[0] # Taking the start token

        for t in range(1, target_length):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            outputs[t] = output
            best = output.argmax(1)
            x = trg[t] if random.random() < teacher_force_ratio else best

        # shape outputs : (target_length, batch_size, output_dimension) so we need to reshape
        outputs = outputs[1:].reshape(-1, outputs.shape[2])
        trg = trg[1:].reshape(-1)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.padTokenIdx)
        H = criterion(outputs, trg)

        return H

    def translateSentence(self, sentence, limit=1000):
        source = [sentence]
        sentence_tensor = self.preparePaddedBatch(source, self.sourceWord2ind)
        device = next(self.parameters()).device

        with torch.no_grad():
            outputs_encoder, hiddens, cells = self.encoder(sentence_tensor)

        outputs = [self.startTokenIdx]

        for _ in range(limit):
            previous = torch.LongTensor([outputs[-1]]).to(device)

            with torch.no_grad():
                output, hiddens, cells = self.decoder(previous, outputs_encoder, hiddens, cells)
                best = output.argmax(1).item()

            outputs.append(best)

            if best == self.endTokenIdx:
                break

        wordset = list(self.targetWord2ind)
        translated = [wordset[prediction] for prediction in outputs[1:-1]]

        return translated
