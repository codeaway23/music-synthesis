import re
import os
import random
import torch

def generate_event():
    if not os.path.exists('./models'):
        os.makedirs('./models')
    models = [os.path.join('./models', x) for x in os.listdir('./models')]
    models = [x for x in models if os.path.isdir(x)]
    event = random.randint(0, len(models)*1000)
    if not os.path.exists(os.path.join('./models', str(event))):
        os.mkdir(os.path.join('./models', str(event)))
    return event

def model_save_utils(generator, discriminator, event, epoch):
    torch.save(generator.state_dict(), os.path.join(
        './models', str(event), 'generator_{}.pt'.format(epoch)))
    torch.save(discriminator.state_dict(), os.path.join(
        './models', str(event), 'discriminator_{}.pt'.format(epoch)))

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def resume_training_util_from_last_epoch(event, generator, discriminator):
    models = natural_sort([os.path.join('./models', event, x) for x in os.listdir('./models')])
    generators = [x for x in models if 'generator' in x]
    discriminators = [x for x in models if 'discriminator' in x]
    latest_gen, latest_disc = generators[-1], discriminators[-1]
    generator = generator.load_state_dict(torch.load(latest_gen))
    discriminator = discriminator.load_state_dict(torch.load(latest_disc))
    epoch = int(latest_gen.split('_')[-1].split('.')[0])
    return generator, discriminator, epoch