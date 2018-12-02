from __future__ import print_function

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

from anomalyDetector import anomalyScore

matplotlib.use('agg')
import matplotlib.pyplot as plt
from rnnmodel import model as mod
import argparse

if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False


def get_ckpt_path(data_dir):
    ckpt_path = 'model_{}.pt'.format(data_dir)
    return ckpt_path


def load_model(data_dir):
    ckpt_path = get_ckpt_path(data_dir)
    if os.path.exists(ckpt_path):
        if use_gpu:
            saved_dict = torch.load(ckpt_path)
        else:
            saved_dict = torch.load(ckpt_path, map_location='cpu')
        model_params = saved_dict['params']
        model = mod.RNNPredictor(rnn_type=model_params['model'],
                                 enc_inp_size=model_params['input_size'],
                                 rnn_inp_size=model_params['emsize'],
                                 rnn_hid_size=model_params['nhid'],
                                 dec_out_size=model_params['input_size'],
                                 nlayers=model_params['nlayers'],
                                 dropout=model_params['dropout'],
                                 tie_weights=model_params['tied'],
                                 res_connection=model_params['res_connection'])
    else:
        model = mod.RNNPredictor(rnn_type=args.model,
                                 enc_inp_size=input_size,
                                 rnn_inp_size=args.emsize,
                                 rnn_hid_size=args.nhid,
                                 dec_out_size=input_size,
                                 nlayers=args.nlayers,
                                 dropout=args.dropout,
                                 tie_weights=args.tied,
                                 res_connection=args.res_connection)

    if os.path.exists(ckpt_path):
        model.load_state_dict(saved_dict['model'])
    return model.double()


def predict(input, data_dir, target=None, future=20, model=None):
    if model is None:
        model = load_model(data_dir)
        input = torch.from_numpy(input).double()
        if target is not None:
            target = torch.from_numpy(target).double().cuda()

        if use_gpu:
            model = model.double().cuda()
            input = input.cuda()
            target = target.cuda()

    model.eval()
    if target is not None:
        with torch.no_grad():
            hidden = model.init_hidden(input.shape[1])
            pred, hidden = model.forward(input, hidden)
            loss = criterion(pred, target)
            print('test loss:', loss.item())
            print(pred[:, 0], target[:, 0])

    outSeq = []
    out = input
    for i in range(future):
        hidden = model.init_hidden(input.shape[1])
        out, hidden = model.forward(out, hidden)
        outSeq.append(out[-1].cpu().detach())
    outSeq = torch.stack(outSeq, 0).numpy()
    scores, rearranged, errors, hiddens, predicted_scores = anomalyScore(3, model, input)
    return scores, outSeq


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data_np_0.npy')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
    parser.add_argument('--augment', type=bool, default=True,
                        help='augment')
    parser.add_argument('--emsize', type=int, default=128,
                        help='size of rnn input 32')
    parser.add_argument('--nhid', type=int, default=128,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=4,
                        help='number of layers')
    parser.add_argument('--res_connection', action='store_true',
                        help='residual connection')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--clip', type=float, default=10,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=400,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=64, metavar='N',
                        help='eval_batch size')
    parser.add_argument('--bptt', type=int, default=50,
                        help='sequence length')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.7,
                        help='teacher forcing ratio (deprecated)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights (deprecated)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='report interval')
    parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                        help='save interval')
    parser.add_argument('--save_fig', action='store_true',
                        help='save figure')
    parser.add_argument('--resume', '-r',
                        help='use checkpoint model parameters as initial parameters (default: False)',
                        action="store_true")
    parser.add_argument('--pretrained', '-p',
                        help='use checkpoint model parameters and do not train anymore (default: False)',
                        action="store_true")
    parser.add_argument('--prediction_window_size', type=int, default=10,
                        help='prediction_window_size')
    args = parser.parse_args()

    data_dir = args.data_dir
    # build the model
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    # data = torch.load('traindata.pt')
    data = np.load(data_dir)
    total_t = data.shape[0]
    input_size = data.shape[1]
    interval = 100
    future_step = 1
    test_size = 50

    input = torch.from_numpy(np.array(
        [data[start:start + interval - 1, :input_size]
         for start in range(total_t - interval - test_size)])).double()
    target = torch.from_numpy(np.array(
        [data[start + 1:start + interval, :input_size]
         for start in range(total_t - interval - test_size)])).double()

    test_input = torch.from_numpy(np.array(
        [data[start:start + interval - 1, :input_size]
         for start in range(total_t - interval - test_size, total_t - interval)])).double()
    test_target = torch.from_numpy(np.array(
        [data[start + 1:start + interval, :input_size]
         for start in range(total_t - interval - test_size, total_t - interval)])).double()
    test_input = test_input.transpose(0, 1)
    test_target = test_target.transpose(0, 1)

    model = load_model(data_dir)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    if use_gpu:
        input = input.cuda()
        target = target.cuda()
        test_input = test_input.cuda()
        test_target = test_target.cuda()
        model = model.double().cuda()
        criterion = criterion.cuda()

    if args.test:
        predict(test_input, data_dir, test_target, model=model)
        exit()

    # train
    for epoch in range(500):
        new_idx = torch.randperm(input.shape[0])
        input = input[new_idx]
        target = target[new_idx]
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        start_time = time.time()
        for i in range(input.shape[0] // args.batch_size):
            inputSeq, targetSeq = \
                input[i * args.batch_size:(i + 1) * args.batch_size].transpose(0, 1), \
                target[i * args.batch_size:(i + 1) * args.batch_size].transpose(0, 1)
            # inputSeq: [ seq_len * batch_size * feature_size ]
            # targetSeq: [ seq_len * batch_size * feature_size ]

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            optimizer.zero_grad()

            '''Loss1: Free running loss'''
            outVal = inputSeq[0].unsqueeze(0)
            outVals = []
            hids1 = []

            hidden = model.init_hidden(args.batch_size)
            for i in range(inputSeq.size(0)):
                outVal, hidden_, hid = model.forward(outVal, hidden, return_hiddens=True)
                outVals.append(outVal)
                hids1.append(hid)
            outSeq1 = torch.cat(outVals, dim=0)
            hids1 = torch.cat(hids1, dim=0)
            loss1 = criterion(outSeq1.reshape(-1), targetSeq.reshape(-1))

            '''Loss2: Teacher forcing loss'''
            hidden = model.init_hidden(args.batch_size)
            outSeq2, hidden, hids2 = model.forward(inputSeq, hidden, return_hiddens=True)
            loss2 = criterion(outSeq2.reshape(-1), targetSeq.reshape(-1))

            '''Loss3: Simplified Professor forcing loss'''
            loss3 = criterion(hids1.reshape(-1), hids2.view(-1).detach())

            '''Total loss = Loss1+Loss2+Loss3'''
            loss = loss1 + loss2 + loss3
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            total_loss += loss2.item()
        print(total_loss / (input.shape[0] // args.batch_size))
        torch.save({
            'model': model.state_dict(), 'params': {
                'model': args.model,
                'input_size': input_size,
                'emsize': args.emsize,
                'nhid': args.nhid,
                'nlayers': args.nlayers,
                'dropout': args.dropout,
                'tied': args.tied,
                'res_connection': args.res_connection,
            }}, get_ckpt_path(args.data_dir))

        test_input[30:35, 0, 0] = 1
        scores, outSeq = predict(test_input[:, 0:1], data_dir, test_target[:, 0:1], model=model)

        # # draw the result
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.plot(np.arange(test_input.size(0)), scores.cpu().numpy(), 'r',
                 linewidth=2.0)
        plt.plot(np.arange(test_input.size(0)), test_input.cpu().numpy()[:, 0, 0], 'b',
                 linewidth=2.0)
        plt.plot(np.arange(test_input.size(0), test_input.size(0) + 20), outSeq[:, 0, 0], 'r:',
                 linewidth=2.0)
        plt.savefig('predict_{}.png'.format(data_dir))
        plt.close()
