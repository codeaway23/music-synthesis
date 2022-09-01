from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from utils import *
from data_utils import *
from constants import *
from dataset import *
from models import *


def generator_discriminator_step(generator, discriminator, criterion, data, resolution, 
        test=False, gen_optimizer=None, disc_optimizer=None, generator_warmup=False):

    noise = get_noise_tensor(data, resolution)
    fake = generator_predict_till_trunc(generator, noise, time_trunc=data.shape[0])

    real_labels = torch.ones((data.shape[1])).to(device)
    fake_labels = torch.zeros(data.shape[1]).to(device)

    if not generator_warmup:
        discriminator.zero_grad(set_to_none=True)
        
        real_D = discriminator(data).view(-1)
        real_disc_loss = criterion(real_D, real_labels)
        if not test:
            real_disc_loss.backward()

        fake_D = discriminator(fake.detach()).view(-1)
        fake_disc_loss = criterion(fake_D, fake_labels)
        if not test:
            fake_disc_loss.backward()

        disc_loss = fake_disc_loss + real_disc_loss
        if not test:
            disc_optimizer.step()

    generator.zero_grad(set_to_none=True)

    fake_D = discriminator(fake).view(-1)
    gen_loss = criterion(fake_D, real_labels)
    if not test:
        gen_loss.backward()
        gen_optimizer.step()
    
    gen_loss = gen_loss.detach().cpu().numpy()
    if not generator_warmup:
        disc_loss = disc_loss.detach().cpu().numpy()
        real_disc_loss = real_disc_loss.detach().cpu().numpy()
        fake_disc_loss = fake_disc_loss.detach().cpu().numpy()
    else:
        disc_loss, real_disc_loss, fake_disc_loss = None, None, None

    if not test:
        return generator, discriminator, gen_loss, disc_loss, real_disc_loss, fake_disc_loss
    else:
        return gen_loss, disc_loss, real_disc_loss, fake_disc_loss


def train(generator, discriminator, criterion, gen_optimizer, disc_optimizer, epochs, train_dataloader, test_dataloader=None, generator_warmup_steps=None, resume=False, event=None, wandb_logs=True):
    gen_train_loss = []
    gen_test_loss = []
    disc_train_loss = []
    disc_test_loss = []
    disc_real_train_loss = []
    disc_real_test_loss = []
    disc_fake_train_loss = []
    disc_fake_test_loss = []
    
    
    if resume == True and event is not None:
        print('EXPERIMENT EVENT: ', event)
        print('resuming training')
        generator, discriminator, res_epoch = resume_training_util_from_last_epoch(event, generator, discriminator)
        
    else:    
        res_epoch = 0
        event = generate_event()
        print('EXPERIMENT EVENT: ', event)
        print('starting training')
    

    print('train dataloader length: ', len(train_dataloader))
    if test_dataloader is not None:
        print('test dataloader length: ', len(test_dataloader))

    if wandb_logs == True:
        wandb.log({
            "event": event,
            "train_dataloader_length": len(train_dataloader),
            "test_dataloader_length": len(test_dataloader) if test_dataloader is not None else None,
            "generator_total_params": sum(p.numel() for p in generator.parameters()),
            "generator_trainable_params": sum(p.numel() for p in generator.parameters() if p.requires_grad),
            "discriminator_total_params": sum(p.numel() for p in discriminator.parameters()),
            "discriminator_trainable_params": sum(p.numel() for p in discriminator.parameters() if p.requires_grad),
            })
        # wandb.watch(generator, log="all")
        # wandb.watch(discriminator, log="all")

    for epoch in range(res_epoch, epochs):
        print('--------------------------------------')
        print('epoch: ', epoch + 1)
        gen_epoch_train_loss = []
        gen_epoch_test_loss = []
        disc_epoch_train_loss = []
        disc_epoch_test_loss = []
        disc_real_epoch_train_loss = []
        disc_real_epoch_test_loss = []
        disc_fake_epoch_train_loss = []
        disc_fake_epoch_test_loss = []

        print('training...')
        generator.train()
        discriminator.train()
        for step, (data, resolution) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            data = data.to(device)
            if generator_warmup_steps is not None and step <= generator_warmup_steps and epoch == 0:
                generator_warmup = True
            else:
                generator_warmup = False
            generator, discriminator, gen_loss, disc_loss, real_disc_loss, fake_disc_loss = generator_discriminator_step(
                generator, discriminator, criterion, data, resolution, gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer, generator_warmup=generator_warmup)

            
            disc_epoch_train_loss.append(disc_loss)
            gen_epoch_train_loss.append(gen_loss)
            disc_real_epoch_train_loss.append(real_disc_loss)
            disc_fake_epoch_train_loss.append(fake_disc_loss)

            if wandb_logs == True:
                wandb.log({
                    "train/epoch/generator_loss": gen_epoch_train_loss[-1], 
                    "train/epoch/discriminator_loss": disc_epoch_train_loss[-1],
                    "train/epoch/discriminator_real_loss": disc_real_epoch_train_loss[-1],
                    "train/epoch/discriminator_fake_loss": disc_fake_epoch_train_loss[-1],
                    "train/epoch/step": epoch * len(train_dataloader) + step
                    })

            del data, resolution, gen_loss, disc_loss, real_disc_loss, fake_disc_loss
        
            torch.cuda.empty_cache()
        
        gen_train_loss.append(np.mean(gen_epoch_train_loss))
        
        disc_train_loss.append(np.mean([x for x in disc_epoch_train_loss if not x == None]))
        disc_real_train_loss.append(np.mean([x for x in disc_real_epoch_train_loss if not x == None]))
        disc_fake_train_loss.append(np.mean([x for x in disc_fake_epoch_train_loss if not x == None]))

        # wandb.log({
        #     "train/generator_loss": gen_train_loss[-1], 
        #     "train/discriminator_loss": disc_train_loss[-1],
        #     "train/discriminator_real_loss": disc_real_train_loss[-1],
        #     "train/discriminator_fake_loss": disc_fake_train_loss[-1],
        #     "train/step": epoch
        #     })
        
        # save models every epoch
        model_save_utils(generator, discriminator, event, epoch)
             
        if test_dataloader:
            print('testing...')
            generator.eval()
            discriminator.eval()
            for step, (data, resolution) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

                data = data.to(device)
                gen_loss, disc_loss, disc_real_loss, disc_fake_loss = generator_discriminator_step(
                    generator, discriminator, criterion, data, resolution, test=True)
                
                gen_epoch_test_loss.append(gen_loss)
                disc_epoch_test_loss.append(disc_loss)
                disc_real_epoch_test_loss.append(disc_real_loss)
                disc_fake_epoch_test_loss.append(disc_fake_loss)
                
                if wandb_logs == True:
                    wandb.log({
                        "test/epoch/generator_loss": gen_epoch_test_loss[-1], 
                        "test/epoch/discriminator_loss": disc_epoch_test_loss[-1],
                        "test/epoch/discriminator_real_loss": disc_real_epoch_test_loss[-1],
                        "test/epoch/discriminator_fake_loss": disc_fake_epoch_test_loss[-1],
                        "test/epoch/step": epoch * len(test_dataloader) + step
                        })

                torch.cuda.empty_cache()
    
            gen_test_loss.append(np.mean(gen_epoch_test_loss))
            disc_test_loss.append(np.mean(disc_epoch_test_loss))
            disc_real_test_loss.append(np.mean(disc_real_epoch_test_loss))
            disc_fake_test_loss.append(np.mean(disc_fake_epoch_test_loss))

            # wandb.log({
            # "test/generator_loss": gen_test_loss[-1],
            # "test/discriminator_loss": disc_test_loss[-1],
            # "test/discriminator_real_loss": disc_real_test_loss[-1],
            # "test/discriminator_fake_loss": disc_fake_test_loss[-1],
            # "test/step": epoch
            # })


def run_sweep():
    default_config = {
        "epochs": 1,
        "batch_size": 6,
        "generator_warmup_steps": None,
        "generator_optimizer": "adam",
        "discriminator_optimizer": "adam",
        "input_size": 128 * 5 + 2,
    }
    with wandb.init(config=default_config) as run:
        
        train_dataset = LPDDataset('data/lpd_5',
                                mode='train', 
                                transform=None)
        test_dataset = LPDDataset('data/lpd_5',
                                mode='test', 
                                transform=None)

        train_dataloader = DataLoader(
                train_dataset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=lambda x: collate_function(x), drop_last=True)
        test_dataloader = DataLoader(
                test_dataset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=lambda x: collate_function(x), drop_last=True)

        criterion = nn.BCELoss()

        generator = RecurrentGeneratorModel(
                wandb.config.input_size, wandb.config.generator_hidden_size, wandb.config.generator_num_layers, activation=wandb.config.generator_activation, dropout=wandb.config.generator_dropout).to(device)
        gen_optimizer = torch.optim.Adam(
                generator.parameters(), lr=wandb.config.generator_learning_rate)

        discriminator = RecurrentDiscriminatorModel(
            wandb.config.input_size, wandb.config.discriminator_hidden_size, wandb.config.discriminator_num_layers, activation=wandb.config.generator_activation, dropout=wandb.config.discriminator_dropout).to(device)
        disc_optimizer = torch.optim.Adam(
                discriminator.parameters(), lr=wandb.config.discriminator_learning_rate)

        train(generator, discriminator, criterion, gen_optimizer,
                disc_optimizer, wandb.config.epochs, train_dataloader, test_dataloader=None, 
                generator_warmup_steps=wandb.config.generator_warmup_steps, resume=False, event=None)
        
        del default_config
        del train_dataset, test_dataset, train_dataloader, test_dataloader
        del generator, discriminator, gen_optimizer, disc_optimizer
        torch.cuda.empty_cache()


def final_train(final_train_config):      
    with wandb.init(config=final_train_config, project="music-gan-sweep", entity="codeaway23") as run:
        
        train_dataset = LPDDataset('data/lpd_5',
                                mode='train', 
                                transform=None)
        test_dataset = LPDDataset('data/lpd_5',
                                mode='test', 
                                transform=None)

        train_dataloader = DataLoader(
                train_dataset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=lambda x: collate_function(x), drop_last=True)
        test_dataloader = DataLoader(
                test_dataset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=lambda x: collate_function(x), drop_last=True)

        criterion = nn.BCELoss()

        generator = RecurrentGeneratorModel(
                wandb.config.input_size, wandb.config.generator_hidden_size, wandb.config.generator_num_layers, activation=wandb.config.generator_activation, dropout=wandb.config.generator_dropout).to(device)
        gen_optimizer = torch.optim.Adam(
                generator.parameters(), lr=wandb.config.generator_learning_rate)

        discriminator = RecurrentDiscriminatorModel(
            wandb.config.input_size, wandb.config.discriminator_hidden_size, wandb.config.discriminator_num_layers, activation=wandb.config.generator_activation, dropout=wandb.config.discriminator_dropout).to(device)
        disc_optimizer = torch.optim.Adam(
                discriminator.parameters(), lr=wandb.config.discriminator_learning_rate)

        train(generator, discriminator, criterion, gen_optimizer,
                disc_optimizer, wandb.config.epochs, train_dataloader, test_dataloader=None, 
                generator_warmup_steps=wandb.config.generator_warmup_steps, resume=False, event=None)
        
        del default_config
        del train_dataset, test_dataset, train_dataloader, test_dataloader
        del generator, discriminator, gen_optimizer, disc_optimizer
        torch.cuda.empty_cache()



if __name__ == '__main__':
    count = 10000
    sweep_id = "codeaway23/music-gan-sweep/hc0qje3r"
    wandb.agent(sweep_id, function=run_sweep, count=count)

    # final_train_config = {
    #   "epochs": 10,
    #   "batch_size": 12,
    #   "generator_warmup_steps": None,
    #   "generator_optimizer": "adam",
    #   "discriminator_optimizer": "adam",
    #   "input_size": 128 * 5,
    #   "generator_learning_rate": 0.01,
    #   "generator_hidden_size": 64,
    #   "generator_num_layers": 4,
    #   "generator_activation": "relu",
    #   "generator_dropout": 0.0,
    #   "discriminator_learning_rate": 0.0001,
    #   "discriminator_hidden_size": 16,
    #   "discriminator_num_layers": 1,
    #   "discriminator_activation": "lrelu",
    #   "discriminator_dropout": 0.2,
    # }

    # final_train(final_train_config)

    # train_dataset = LPDDataset('data/lpd_5',
    #                         mode='train', 
    #                         transform=None,
    #                         trunc=128*3)
    # test_dataset = LPDDataset('data/lpd_5',
    #                         mode='test', 
    #                         transform=None,
    #                         trunc=128*3)

    # train_dataloader = DataLoader(
    #         train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: collate_function(x), drop_last=True)
    # test_dataloader = DataLoader(
    #         test_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: collate_function(x), drop_last=True)

    # criterion = nn.BCELoss()

    # generator = RecurrentGeneratorModel(
    #         642, 8, 3, activation='relu', dropout=0.).to(device)
    # gen_optimizer = torch.optim.Adam(
    #         generator.parameters(), lr=0.001)

    # discriminator = RecurrentDiscriminatorModel(
    #     642, 8, 3, activation='lrelu', dropout=0.2).to(device)
    # disc_optimizer = torch.optim.Adam(
    #         discriminator.parameters(), lr=0.0001)

    # train(generator, discriminator, criterion, gen_optimizer,
    #     disc_optimizer, 2, train_dataloader, test_dataloader=None, 
    #     generator_warmup_steps=None, resume=False, event=None, wandb_logs=False)
