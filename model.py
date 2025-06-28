import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os 

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

class Corpus:
    def __init__(self, path, batch_size, max_sql):
        self.vocabulary = []
        self.word_id = {}
        self.train = self.tokenize(os.path.join(path, 'data', '1000.fasta'))
        self.valid = self.tokenize(os.path.join(path, 'data', '1000_1250.fasta'))
        self.dset_flag = 'train'
        self.train_si = 0
        self.valid_si = 0
        self.max_sql = max_sql
        self.batch_size = batch_size
        print('size of train set:', self.train.size(0))
        print('size of valid set:', self.valid.size(0))
        self.trian_batch_num = self.train.size(0) // self.batch_size['train']
        self.valid_batch_num = self.valid.size(0) // self.batch_size['valid']
        self.train = self.train.narrow(0,0,self.batch_size['train'] * self.trian_batch_num)
        self.valid = self.valid.narrow(0,0,self.batch_size['valid'] * self.valid_batch_num)
        self.train = self.train.view(self.batch_size['train'], -1).t().contiguous()
        self.valid = self.valid.view(self.batch_size['valid'], -1).t().contiguous()

    def tokenize(self, file_name):
        with open(file_name, 'r', encoding='utf-8') as file:
            file_lines = file.readlines()
            sequence_lines = []
            num_of_words = 0
            for line in file_lines:
                if line.startswith('>'):
                    continue
                words = list(line.strip()) + ['<eos>']
                num_of_words += len(words)
                sequence_lines.append(words)
                for word in words:
                    if word not in self.vocabulary:
                        self.word_id[word] = len(self.vocabulary)
                        self.vocabulary.append(word)
            file_tokens = torch.LongTensor(num_of_words)
            token_id = 0 
            for sequence in sequence_lines:
                for word in sequence:
                    file_tokens[token_id] = self.word_id[word]
                    token_id += 1
            return file_tokens           
        
    def set_train(self):    
        self.dset_flag = 'train'
        self.si = 0
    
    def set_valid(self):
        self.dset_flag = 'valid'
        self.si = 0    
        
    def get_batch(self, device=None):
        if self.dset_flag == 'train':
            start_index = self.train_si
            seq_len = min(self.max_sql, self.train.size(0)-self.train_si-1)
            data_loader = self.train
            self.train_si += seq_len
        else:
            start_index = self.valid_si
            seq_len = min(self.max_sql, self.valid.size(0)-self.valid_si-1)
            data_loader = self.valid
            self.valid_si += seq_len
        data = data_loader[start_index:start_index+seq_len, :]
        target = data_loader[start_index+1:start_index+seq_len+1, :].view(-1)

        if self.dset_flag == 'train' and self.train_si+1 == self.train.size(0):
            end_flag = True
            self.train_si = 0
        elif self.dset_flag == 'valid' and self.valid_si+1 == self.valid.size(0):
            end_flag = True
            self.valid_si = 0
        else:
            end_flag = False
        
        if device is not None:
            data = data.to(device)
            target = target.to(device)
        
        return data, target, end_flag
    
class RNN(nn.Module):
    def __init__(self, nvoc, ninput, nhid, nlayers):
        super(RNN, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.embed = nn.Embedding(nvoc, ninput)
        self.gru = nn.GRU(ninput, nhid, nlayers)
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.nlayers = nlayers
        self.nhid = nhid

    def init_weights(self):
        inint_uniform = 0.1
        self.embed.weight.data.uniform_(-inint_uniform, inint_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-inint_uniform, inint_uniform)

    def forward(self, input):
        embeddings = self.drop(self.embed(input))
        output, hidden = self.gru(embeddings)
        output = self.drop(output)
        decoded = self.decoder(output.reshape(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

parser = argparse.ArgumentParser(description='GRU Practice')
parser.add_argument('--epochs', type=int, default=200, help='upper limit of epochs')
parser.add_argument('--train_batch_size', type=int, default=6400, metavar='N', help='batch size for training')
parser.add_argument('--eval_batch_size', type=int, default=3200, metavar='N', help='batch size for evaluation')
parser.add_argument('--max_sql', type=int, default=35, metavar='N', help='max length of sql')
parser.add_argument('--cuda', action='store_true', help='use cuda device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
parser.add_argument('--seed', type=int, default=214, help='random seed for initialization')
args = parser.parse_args()

torch.manual_seed(args.seed)

train_batch_size = args.train_batch_size
ecal_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size, 'valid': ecal_batch_size}
cur_dir = os.path.dirname(os.path.abspath(__file__))
print('Current directory:', cur_dir)
data_loader = Corpus(cur_dir, batch_size, args.max_sql)

nvoc = len(data_loader.vocabulary)
ninput = 128  # Size of the input embedding
nhid = 256  # Size of the hidden state
nlayers = 2  # Number of GRU layers

net = RNN(nvoc, ninput, nhid, nlayers)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def train():
    net.train()
    total_loss = 0
    for batch, i in enumerate(range(0, data_loader.train.size(0) - 1, args.max_sql)):
        data, targets, _ = data_loader.get_batch()
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output, hidden = net(data)
        loss = criterion(output.view(-1, nvoc), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / (i + 1)

def evaluate():
    net.eval()
    total_loss = 0
    with torch.no_grad():
        for batch, i in enumerate(range(0, data_loader.valid.size(0) - 1, args.max_sql)):
            data, targets, _ = data_loader.get_batch()
            data, targets = data.to(device), targets.to(device)
            output, hidden = net(data)
            loss = criterion(output.view(-1, nvoc), targets)
            total_loss += loss.item()
    return total_loss / (i + 1)

# train_losses = []
# valid_losses = []
# train_perplexities = []
# valid_perplexities = []

# for epoch in range(args.epochs):
#     train_loss = train()
#     valid_loss = evaluate()
#     train_losses.append(train_loss)
#     valid_losses.append(valid_loss)
#     train_perplexities.append(np.exp(train_loss))
#     valid_perplexities.append(np.exp(valid_loss))
#     print(f'Epoch {epoch},  Training Loss: {train_loss:.4f}, Validtion Loss: {valid_loss:.4f}'
#         f', Training Perplexity: {np.exp(train_loss):.4f}, Validation Perplexity: {np.exp(valid_loss):.4f}')
    

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Training Loss')
# plt.plot(valid_losses, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')

# plt.subplot(1, 2, 2)
# plt.plot(train_perplexities, label='Training Perplexity')
# plt.plot(valid_perplexities, label='Validation Perplexity')
# plt.title('Training and Validation Perplexity')
# plt.xlabel('Epochs')
# plt.ylabel('Perplexity')
# plt.legend()

# plt.show()

# torch.save(net.state_dict(), 'protein_model.path')

# def predict(initial_sequence, num_residues, unknown_token='<eos>'):
#     net.eval()
#     residues = initial_sequence.copy()
#     for _ in range(num_residues):
#         input_indices = [data_loader.word_id.get(residue, data_loader.word_id[unknown_token]) for residue in residues[-args.max_sql:]]
#         input_tensor = torch.LongTensor(input_indices).view(-1,1).to(device)
#         with torch.no_grad():
#             output, _ = net(input_tensor)
#         last_residue_logits = output[-1]
#         predicted_residue_id = torch.argmax(last_residue_logits).item()
#         predicted_residue = data_loader.vocabulary[predicted_residue_id]
#         residues.append(predicted_residue)
#     return residues

# initial_sequence = ['M', 'E', 'N', 'S', 'D']
# predicted_sequence = predict(initial_sequence, 20)
# print(''.join(predicted_sequence))

def predict(initial_sequence, num_residues, net, data_loader, device, max_sql, unknown_token='<eos>'):
    """
    Predict the next num_residues residues given an initial sequence.
    Args:
        initial_sequence (list): List of residue characters as input.
        num_residues (int): Number of residues to predict.
        net (nn.Module): Trained model.
        data_loader (Corpus): Corpus object with vocabulary and word_id.
        device (torch.device): Device to run prediction on.
        max_sql (int): Maximum sequence length for context window.
        unknown_token (str): Token for unknown residues.
    Returns:
        list: The initial sequence plus predicted residues.
    """
    net.eval()
    residues = initial_sequence.copy()
    for _ in range(num_residues):
        input_indices = [
            data_loader.word_id.get(residue, data_loader.word_id[unknown_token])
            for residue in residues[-max_sql:]
        ]
        input_tensor = torch.LongTensor(input_indices).view(-1, 1).to(device)
        with torch.no_grad():
            output, _ = net(input_tensor)
        last_residue_logits = output[-1]
        predicted_residue_id = torch.argmax(last_residue_logits).item()
        predicted_residue = data_loader.vocabulary[predicted_residue_id]
        residues.append(predicted_residue)
    return residues

net = RNN(nvoc, ninput, nhid, nlayers)
net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'protein_model.path'), map_location=device))  # 加载参数
net.to(device)


initial_sequence = ['M', 'E', 'N', 'S', 'D']
predicted_sequence = predict(
    initial_sequence,
    num_residues=20,
    net=net,
    data_loader=data_loader,
    device=device,
    max_sql=args.max_sql
)
print(''.join(predicted_sequence))


