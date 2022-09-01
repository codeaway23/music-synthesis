import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import pypianoroll

from constants import *

def get_noise_tensor(data, resolution):
    noise_shape = (max(resolution) * 4, data.shape[1], data.shape[2])
    noise = torch.zeros(noise_shape).to(device)
    for i, res in enumerate(resolution):
        noise[:res * 4, i, :] = data[:res * 4, i, :]
    return noise

def generator_predict_till_trunc(generator, input, time_trunc):
    for i in range(input.shape[0], time_trunc):
        generator.zero_grad(set_to_none=True)
        input = torch.cat(
            (input, generator(input).detach().unsqueeze(0)), dim=0)
        torch.cuda.empty_cache()
    input = (input > 0.5).float() * 1
    return input

def collate_function(data):
    resolution = [x[1] for x in data]
    data = [x[0] for x in data]
    data = pad_sequence(data)
    return data, resolution


def save_pianoroll_midi(pr, event, i, mode="pred", resolution=None):
    if mode == 'target':
        path = os.path.join('preds', str(event), 'targets', 'midi_{}.mid'.format(i))
    elif mode == 'pred':
        path = os.path.join('preds', str(event), 'preds', 'midi_{}.mid'.format(i))
    if not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))
    pypianoroll.write(path, pr)
    