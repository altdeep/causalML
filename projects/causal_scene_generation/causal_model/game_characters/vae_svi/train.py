import argparse
import numpy as np
import torch
import torch.nn as nn
import visdom
import os
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from utils.game_character_dataloader import GameCharacterFullData, setup_data_loaders
from torchvision import transforms,models
import time
from models.causal_vae import VAE

def train(args, DATA_PATH):
    # clear param store
    pyro.clear_param_store()
    #pyro.enable_validation(True)

    # train_loader, test_loader
    transform = {}
    transform["train"] = transforms.Compose([
                                    transforms.Resize((400,400)),
                                    transforms.ToTensor(),
                                ])
    transform["test"] = transforms.Compose([
                                    transforms.Resize((400,400)),
                                    transforms.ToTensor()
                                ])

    train_loader, test_loader = setup_data_loaders(dataset=GameCharacterFullData, root_path = DATA_PATH, batch_size=32, transforms=transform)

    # setup the VAE
    vae = VAE(use_cuda=args.cuda, num_labels = 17)

    # setup the exponential learning rate scheduler
    optimizer = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': {'lr': args.learning_rate}, 'gamma': 0.1})


    # setup the inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, scheduler, loss=elbo)

   
     # setup visdom for visualization
    if args.visdom_flag:
        vis = visdom.Visdom(port='8097')

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, y, actor, reactor, actor_type, reactor_type, action, reaction in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
                actor = actor.cuda()
                reactor = reactor.cuda()
                actor_type = actor_type.cuda()
                reactor_type = reactor_type.cuda()
                action = action.cuda()
                reaction = reaction.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x,y, actor,reactor, actor_type,reactor_type, action, reaction)

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.
            # compute the loss over the entire test set
            for i, (x, y, actor, reactor, actor_type, reactor_type, action, reaction) in enumerate(test_loader):
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                    actor = actor.cuda()
                    reactor = reactor.cuda()
                    actor_type = actor_type.cuda()
                    reactor_type = reactor_type.cuda()
                    action = action.cuda()
                    reaction = reaction.cuda()
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x,y, actor,reactor, actor_type,reactor_type, action, reaction)
                # pick three random test images from the first mini-batch and
                # visualize how well we're reconstructing them
                if i == 0:
                    if args.visdom_flag:
                        plot_vae_samples(vae, vis)
                        reco_indices = np.random.randint(0, x.shape[0], 3)
                        for index in reco_indices:
                            test_img = x[index, :]
                            reco_img = vae.reconstruct_img(test_img)
                            vis.image(test_img.reshape(400, 400).detach().cpu().numpy(),
                                      opts={'caption': 'test image'})
                            vis.image(reco_img.reshape(400, 400).detach().cpu().numpy(),
                                      opts={'caption': 'reconstructed image'})
            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo.append(total_epoch_loss_test)
            print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))
    
    return vae, optimizer

if __name__ == '__main__':
    assert pyro.__version__.startswith('1.3.1')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=101, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    parser.add_argument('-visdom', '--visdom_flag', action="store_true", help='Whether plotting in visdom is desired')
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    GDRIVE_DATA_PATH = "gdrive/My Drive/causal_scene_generation/vae_svi/data/"
    LOCAL_DATA_PATH = "./data/"

    GDRIVE_MODEL_PATH = "gdrive/My Drive/causal_scene_generation/vae_svi/model/"
    LOCAL_MODEL_PATH = "./model/"


    DATA_PATH = GDRIVE_DATA_PATH if 'COLAB_GPU' in os.environ else LOCAL_DATA_PATH
    MODEL_PATH = GDRIVE_MODEL_PATH if 'COLAB_GPU' in os.environ else LOCAL_MODEL_PATH

    model, optimizer = train(args, DATA_PATH)
    t = time.time()
    torch.save(model.state_dict(), MODEL_PATH+"vae_model"+str(t)+".pkl")
    scheduler.save(MODEL_PATH+"optimizer"+str(t)+".pkl")