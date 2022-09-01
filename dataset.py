import glob
import pypianoroll
import torch
from torch import tensor, zeros
from torch.utils.data import Dataset

from constants import *

class LPDDataset(Dataset):
    def __init__(self, root_dir, trunc=24*4*5, split=0.2, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.time_trunc = trunc
        self.files = self.recursive_list_of_files()
        self.train = self.files[:int(len(self.files) * (1 - split))]
        self.test = self.files[int(len(self.files) * (1 - split)):]
        

    def recursive_list_of_files(self, ):
        files = glob.glob(self.root_dir + '/**/*.npz', recursive=True)
        # shuffle(files)
        return files


    def midi_content_to_tensor(self, data, tempo, downbeat):
        time = tempo.shape[0]
        tempo = tempo.unsqueeze(0)
        downbeat = downbeat.unsqueeze(0)
        final = []
        for track in data.tracks:
            array = tensor(track.pianoroll).to(device)
            if array.shape[0] == 0:
                array = zeros((time, 128)).to(device)
            final.append(array)
        final = torch.cat(final, 1)
        final = torch.cat([tempo, downbeat, final.transpose(0,1)])
        final = final.transpose(1,0)
        final = final[:self.time_trunc, :]
        final = final.type(torch.FloatTensor)
        del array, data
        return final

    def tensor_content_to_midi(self, tensor, resolution):
        time = tensor.shape[0]
        tempo = tensor[:, 0]
        downbeat = tensor[:, 1]
        tensor = tensor[:, 2:]
        programs = [0, 0, 24, 32, 48]  # program number for each track
        is_drums = [True, False, False, False, False] # drum indicator for each track
        track_names = ['Drums', 'Piano', 'Guitar',
                    'Bass', 'Strings']  # name of each track
        tracks = []
        for i in range(len(is_drums)):
            pr = tensor[:, 128*(i):128*(i+1)].detach().cpu().numpy()
            pr = np.float64(pr)
            if np.count_nonzero(pr) == 0:
                pr = np.zeros((0,128))

            if not i == 0:
                tracks.append(
                    pypianoroll.StandardTrack(
                    name=track_names[i],
                    is_drum=is_drums[i],
                    program=programs[i],
                    pianoroll=pr
                )
            )
            else:
                tracks.append(
                    pypianoroll.BinaryTrack(
                    name=track_names[i],
                    is_drum=is_drums[i],
                    program=programs[i],
                    pianoroll=pr
                )
            )
        downbeat = downbeat.detach().cpu().numpy()
        downbeat = np.float64(downbeat)        
        tempo = tempo.detach().cpu().numpy()
        tempo = np.float64(tempo)
        return pypianoroll.Multitrack(name='test', tracks=tracks, tempo=tempo, downbeat=downbeat, resolution=resolution)


    def __len__(self):
        if self.mode == 'train':
            return len(self.train)
        elif self.mode == 'test':
            return len(self.test)


    def __getitem__(self, idx,):
        if self.mode == 'train':
            file = self.train[idx]
        elif self.mode == 'test':
            file = self.test[idx]
        data = pypianoroll.load(file)
        resolution = data.resolution
        time = data.tempo.shape[0]
        tempo = torch.tensor(data.tempo).to(device)
        downbeat = torch.tensor(data.downbeat).to(device)
        data = self.midi_content_to_tensor(data, tempo, downbeat)
        if self.transform:
            data = self.transform(data)
        return data, resolution



if __name__ == '__main__':
    train = LPDDataset('data/lpd_5',
                    mode='train', 
                    transform=None,
                    trunc=None)
    d, r = train[1]
    m = train.tensor_content_to_midi(d, r)
    pypianoroll.write('x.mid', m)
    
