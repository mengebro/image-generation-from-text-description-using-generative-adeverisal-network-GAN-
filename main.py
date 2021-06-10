from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from DAMSM import RNN_ENCODER ,CNN_ENCODER, BERT_CNN_ENCODER_RNN_DECODER,BERT_RNN_ENCODER,CNN_ENCODER_RNN_DECODER
from pytorch_pretrained_bert import BertModel
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import NetG,NetD
import torchvision.utils as vutils
from miscc.loss import image_to_text_loss
from pytorch_pretrained_bert import BertModel
#from miscc.losses import words_loss, cycle_generator_loss

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

import multiprocessing
multiprocessing.set_start_method('spawn', True)


UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='E:/menge/DF-GAN-cycle/cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def sampling(text_encoder, netG, dataloader,device):
    
    model_dir = cfg.TRAIN.NET_G
    split_dir = 'valid'
    # Build and load the generator
    netG.load_state_dict(torch.load(r'E:\menge\DF-GAN-ATTN\model\bird\netG_114.pth.tar'))
    netG.eval()
    #addd the loading path .pth.tar

    batch_size = cfg.TRAIN.BATCH_SIZE
    s_tmp = model_dir
    save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(save_dir)
    cnt = 0
    for i in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        for step, data in enumerate(dataloader, 0):
            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            cnt += batch_size
            if step % 100 == 0:
                print('step: ', step)
            # if step > 50:
            #     break
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            ########################################################################
            
            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            ###########################################################################
            #######################################################
            # (2) Generate fake images
            ######################################################
            with torch.no_grad():
                noise = torch.randn(batch_size, 100)
                noise=noise.to(device)
                fake_imgs = netG(noise,sent_emb,words_embs,mask)
            for j in range(batch_size):
                s_tmp = '%s/single/%s' % (save_dir, keys[j])
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    print('Make a new folder: ', folder)
                    mkdir_p(folder)
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                fullpath = '%s_%3d.png' % (s_tmp,i)
                im.save(fullpath)



def train(dataloader,netG,netD,text_encoder,optimizerG,optimizerD,state_epoch,batch_size,device,image_encoder):
    for epoch in range(state_epoch+1, cfg.TRAIN.MAX_EPOCH+1):
        for step, data in enumerate(dataloader, 0):
            
            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            
            ########################################################################
            
            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            ###########################################################################

            imgs=imags[0].to(device)
            real_features = netD(imgs)
            output = netD.COND_DNET(real_features,sent_emb)
            errD_real = torch.nn.ReLU()(1.0 - output).mean()

            output = netD.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
            errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()

            # synthesize fake images
            noise = torch.randn(batch_size, 100)
            noise=noise.to(device)
            fake = netG(noise,sent_emb,words_embs,mask)  
            #to get the feature of the generated image 
            region_features, cnn_code, word_logits = image_encoder(fake, captions)

            # G does not need update with D
            fake_features = netD(fake.detach()) 

            errD_fake = netD.COND_DNET(fake_features,sent_emb)
            errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()          

            errD = errD_real + (errD_fake + errD_mismatch)/2.0
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            errD.backward()
            optimizerD.step()

            #MA-GP
            interpolated = (imgs.data).requires_grad_()
            sent_inter = (sent_emb.data).requires_grad_()
            features = netD(interpolated)
            out = netD.COND_DNET(features,sent_inter)
            grads = torch.autograd.grad(outputs=out,
                                    inputs=(interpolated,sent_inter),
                                    grad_outputs=torch.ones(out.size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            grad = torch.cat((grad0,grad1),dim=1)                        
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm) ** 8)
            d_loss = 4.0 * d_loss_gp
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            d_loss.backward()
            optimizerD.step()
            ##########################################
            # adding cycle
            t_loss = image_to_text_loss(word_logits, captions) * 1.0
            ###########################################

            # update G
            features = netD(fake)
            output = netD.COND_DNET(features,sent_emb)
            #addd the cycel loss 
            errG = output + t_loss
            errG = - errG.mean()
            #errG = torch.mean(errG)
            torch.cuda.empty_cache()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
                % (epoch, cfg.TRAIN.MAX_EPOCH, step, len(dataloader), errD.item(), errG.item()))

        vutils.save_image(fake.data,
                        '%s/fake_samples_epoch_%03d.png' % ('E:/menge/DF-GAN-ATTN/fake_imgs/', epoch),
                        normalize=True)

        if epoch%2==0:
            torch.save(netG.state_dict(), 'E:/menge/DF-GAN-ATTN/model/%s/netG_%03d.pth.tar' % (cfg.CONFIG_NAME, epoch))
            torch.save(netD.state_dict(), 'E:/menge/DF-GAN-ATTN/model/%s/netD_%03d.pth.tar' % (cfg.CONFIG_NAME, epoch))       
            # torch.save(netG.state_dict(), 'E:/menge/DF-GAN with Cycle coco/models/%s/netG_%03d.pth' % (cfg.CONFIG_NAME, epoch))
            # torch.save(netD.state_dict(), 'E:/menge/DF-GAN with Cycle coco/models/%s/netD_%03d.pth' % (cfg.CONFIG_NAME, epoch))
            save_path = 'E:/menge/DF-GAN-ATTN/model/checkpoint.pth'
            print('saving the model at the end of epoch %d, iters %d' % (epoch, step))        
            #save_checkpoint(netG, optimizerG, save_path, epoch, step)
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'netG': netG.state_dict(),
                'netD': netD.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict(),
            }
            torch.save(checkpoint, save_path)       
        
        if step % 100 == 0:
           
            log.info(
                "Iter %d: gen_loss=%.3e, dis_loss=%.3e",
                iter_no,
                np.mean(errG),
                np.mean(errD),)
            writer.add_scalar("gen_loss", np.mean(errG), step)
            writer.add_scalar("dis_loss", np.mean(errD), step)
            writer.flush()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    return count




if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
        #args.manualSeed = random.randint(1, 10000)
    print("seed now is : ",args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = 'E:/menge/DF-GAN-ATTN/output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    torch.cuda.set_device(cfg.GPU_ID)
   #torch.cuda.set_device(0)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.B_VALIDATION:
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                                base_size=cfg.TREE.BASE_SIZE,
                                transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:     
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = NetG(cfg.TRAIN.NF, 100).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)
    ####################################################
    # checkpoint = torch.load(r'E:\menge\DF-GAN-cycle\models\bird\netD_526.pth.tar')
    # netG.load_state_dict(checkpoint)
    # # netG.load_state_dict(torch.load(r'E:\menge\DF-GAN-cycle\models\bird\netD_526.pth.tar'))
    # netG.eval()
    ##################################################

    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False
    #text_encoder.eval()  

    image_encoder = CNN_ENCODER_RNN_DECODER(768, 256,
                                            dataset.n_words, rec_unit='LSTM')

    # print("the model CNN_ENCODER_RNN_DECODER")
    # print(image_encoder)
    img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
    state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    image_encoder.load_state_dict(state_dict, strict=False)

    # try:
    #     image_encoder.load_state_dict(state_dict, strict=False)
    # except RuntimeError as e:
    #     print('Ignoring "' + str(e) + '"')

    for p in image_encoder.parameters():
        p.requires_grad = False
    print('Load image encoder from:', img_encoder_path)
    #image_encoder.eval()

    print("the shape of image_encoder")
    print(image_encoder)

    image_encoder = image_encoder.cuda()

    state_epoch=0

    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))  
    torch.cuda.empty_cache()

    if cfg.B_VALIDATION:
        count = sampling(text_encoder, netG, dataloader,device)  # generate images for the whole valid dataset
        print('state_epoch:  %d'%(state_epoch))
    else:
        # torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        # load_path = 'E:/menge/DF-GAN-ATTN/model/checkpoint.pth'
        # loaded_checkpoint = torch.load(load_path)
        # state_epoch = loaded_checkpoint['epoch']
        # step = loaded_checkpoint['step']
        # #netG = netG() # instantiate your model
        # #netD = netD() # instantiate your model
        # netG.load_state_dict(loaded_checkpoint['netG'])
        # netD.load_state_dict(loaded_checkpoint['netD'])
        # #optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    
        # #optimizerG = torch.optim.SGD(optimizerG.parameters(), lr=0, momentum=0) # or whatever optimizer you use
        # optimizerG.load_state_dict(loaded_checkpoint['optimizerG'])
        # #optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9)) 
        # #optimizerD = torch.optim.SGD(optimizerD.parameters(), lr=0, momentum=0) # or whatever optimizer you use
        # optimizerD.load_state_dict(loaded_checkpoint['optimizerD'])

        # torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        count = train(dataloader,netG,netD,text_encoder,optimizerG,optimizerD, state_epoch,batch_size,device,image_encoder)



        