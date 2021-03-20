import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from .encoder import Encoder
from .decoder import Decoder
from utils.utils import return_cpts, return_values, return_inverse_cpts




class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 500 hidden units
    def __init__(self, z_dim=128, hidden_dim=1024, use_cuda=False, num_labels=17):
        super().__init__()
        self.output_size = num_labels
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim, num_labels)
        self.decoder = Decoder(z_dim, hidden_dim, num_labels) # 3 channel image.
        self.values = return_values()
        self.cpts = return_cpts()
        self.inverse_cpts = return_inverse_cpts()


        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x,y, actorObs, reactorObs, actor_typeObs, reactor_typeObs, actionObs, reactionObs):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        options = dict(dtype=x.dtype, device=x.device)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            
            # decode the latent code z
            # The label y  is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)

            #print(f"In model actor is {actorObs}, reactor is {reactorObs}, actor_type is {actor_typeObs} and reactor_type is {reactor_typeObs}")
            '''
            Causal Model
            '''

            '''
            The below should basically be a concatenation of actor's action and reactor's reaction.
            '''

            actor = pyro.sample("actor", dist.OneHotCategorical(self.cpts["character"]), obs=actorObs).cuda()
            act_idx = actor[..., :].nonzero()[:, 1].cuda()

            reactor = pyro.sample("reactor", dist.OneHotCategorical(self.cpts["character"]), obs=reactorObs).cuda()
            rct_idx = reactor[..., :].nonzero()[:, 1].cuda()


            # To choose the type of Satyr or Golem (type 1, 2 or 3. This translates to different image of that character.)
            actor_type = pyro.sample("actor_type", dist.OneHotCategorical(self.cpts["type"][act_idx]), obs=actor_typeObs).cuda()
            act_typ_idx = actor_type[..., :].nonzero()[:, 1].cuda()

            reactor_type = pyro.sample("reactor_type", dist.OneHotCategorical(self.cpts["type"][rct_idx]), obs=reactor_typeObs).cuda()
            rct_typ_idx = reactor_type[..., :].nonzero()[:, 1].cuda()


            # To choose the strength, defense and attack based on the character and its type. Either Low or High
            actor_strength = pyro.sample("actor_strength", dist.Categorical(self.cpts["strength"][act_idx, act_typ_idx])).cuda()
            actor_defense = pyro.sample("actor_defense", dist.Categorical(self.cpts["defense"][act_idx, act_typ_idx])).cuda()
            actor_attack = pyro.sample("actor_attack", dist.Categorical(self.cpts["attack"][act_idx, act_typ_idx])).cuda()

            # To choose the character's(actor, who starts the fight) action based on the strength, defense and attack capabilities
            actor_action = pyro.sample("actor_action", dist.OneHotCategorical(self.cpts["action"][actor_strength, actor_defense, actor_attack]), obs=actionObs).cuda()

            # Converting onehot categorical to categorical value
            sampled_actor_action = actor_action[..., :].nonzero()[:, 1].cuda()
            # To choose the other character's strength, defense and attack based on the character and its type
            reactor_strength = pyro.sample("reactor_strength", dist.Categorical(self.cpts["strength"][rct_idx, rct_typ_idx])).cuda()
            reactor_defense = pyro.sample("reactor_defense", dist.Categorical(self.cpts["defense"][rct_idx, rct_typ_idx])).cuda()
            reactor_attack = pyro.sample("reactor_attack", dist.Categorical(self.cpts["attack"][rct_idx, rct_typ_idx])).cuda()

            # To choose the character's (reactor, who reacts to the actor's action in a duel) reaction based on its own strength, defense , attack and the other character's action.
            reactor_reaction = pyro.sample("reactor_reaction", dist.OneHotCategorical(self.cpts["reaction"][reactor_strength, reactor_defense, reactor_attack, sampled_actor_action]), obs=reactionObs).cuda()

            #Modiying actor/reactor type tensor sizes to match the original num_labels.

            #actor_type = modify_type_tensor(actor_type, act_idx)
            #reactor_type = modify_type_tensor(reactor_type, rct_idx)

            ys = torch.cat([actor, actor_type, actor_action, reactor, reactor_type, reactor_reaction], dim=-1).cuda()

            '''
            Basically, the following should be a concatenation of actor's action and reactor's reaction
            '''
            #alpha_prior = torch.ones(x.shape[0], self.output_size, **options) / (1.0 * self.output_size)
            #ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            loc_img = self.decoder.forward(z,ys)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(3), obs=x)

            #print(f"actor is {actor},reactor is {reactor}, actor_type is {actor_type}, reactor_type is {reactor_type},actor_strength is {actor_strength}, actor_defense is {actor_defense},actor_attack is {actor_attack}, actor_action is {actor_action}, sampled_actor_action is {sampled_actor_action}, reactor_strength is {reactor_strength}, reactor_attack is {reactor_attack}, reactor_defense is {reactor_defense},reactor_reaction is {reactor_reaction}, ys is {ys}")
            # return the loc so we can visualize it later
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, y, actorObs, reactorObs, actor_typeObs, reactor_typeObs, actionObs, reactionObs):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            act_idx = actorObs[..., :].nonzero()[:, 1].cuda()
            rct_idx = reactorObs[..., :].nonzero()[:, 1].cuda()
            act_typ_idx = actor_typeObs[..., :].nonzero()[:, 1].cuda()
            rct_typ_idx = reactor_typeObs[..., :].nonzero()[:, 1].cuda()
            action, reaction = torch.nonzero(actionObs)[:, 1].cuda(), torch.nonzero(reactionObs)[:, 1].cuda()
            # use the encoder to get the parameters used to define q(z|x)
            actor_strength = pyro.sample("actor_strength", dist.Categorical(self.inverse_cpts["action_strength"][action, act_typ_idx, act_idx])).cuda()
            actor_defense = pyro.sample("actor_defense", dist.Categorical(self.inverse_cpts["action_defense"][action, act_typ_idx, act_idx])).cuda()
            actor_attack = pyro.sample("actor_attack", dist.Categorical(self.inverse_cpts["action_attack"][action, act_typ_idx, act_idx])).cuda()

            reactor_strength = pyro.sample("reactor_strength", dist.Categorical(self.inverse_cpts["reaction_strength"][reaction, rct_typ_idx, rct_idx])).cuda()
            reactor_defense = pyro.sample("reactor_defense", dist.Categorical(self.inverse_cpts["reaction_defense"][reaction, rct_typ_idx, rct_idx])).cuda()
            reactor_attack = pyro.sample("reactor_attack", dist.Categorical(self.inverse_cpts["reaction_attack"][reaction, rct_typ_idx, rct_idx])).cuda()


            #print(f"actor is {actorObs}, reactor is {reactorObs}, actor_type is {actor_typeObs}, reactor_type is {reactor_typeObs}, actor_strength is {actor_strength}, actor_defense is {actor_defense}, actor_attack is {actor_attack}, actor_action is {action},reactor_strength is {reactor_strength},reactor_attack is {reactor_attack},reactor_defense is {reactor_defense}, reactor_reaction is {reaction}")

            z_loc, z_scale = self.encoder.forward(x,y) # y -> action and reaction
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
    
    def inference_model(self):
        
        # Causal model - DAG implementation
        actor = pyro.sample("actor", dist.OneHotCategorical(self.cpts["character"])).cuda()
        act_idx = actor[..., :].nonzero()[:, 0].cuda()

        reactor = pyro.sample("reactor", dist.OneHotCategorical(self.cpts["character"])).cuda()
        rct_idx = reactor[..., :].nonzero()[:, 0].cuda()

        # To choose the type of Satyr or Golem (type 1, 2 or 3. This translates to different image of that character.)
        actor_type = pyro.sample("actor_type", dist.OneHotCategorical(self.cpts["type"][act_idx])).cuda()
        act_typ_idx = actor_type[..., :].nonzero()[:, 0].cuda()

        reactor_type = pyro.sample("reactor_type", dist.OneHotCategorical(self.cpts["type"][rct_idx])).cuda()
        rct_typ_idx = reactor_type[..., :].nonzero()[:, 0].cuda()

        # To choose the strength, defense and attack based on the character and its type. Either Low or High
        actor_strength = pyro.sample("actor_strength", dist.Categorical(self.cpts["strength"][act_idx, act_typ_idx])).cuda()
        actor_defense = pyro.sample("actor_defense", dist.Categorical(self.cpts["defense"][act_idx, act_typ_idx])).cuda()
        actor_attack = pyro.sample("actor_attack", dist.Categorical(self.cpts["attack"][act_idx, act_typ_idx])).cuda()

        # To choose the character's(actor, who starts the fight) action based on the strength, defense and attack capabilities
        actor_action = pyro.sample("actor_action", dist.OneHotCategorical(self.cpts["action"][actor_strength, actor_defense, actor_attack])).cuda()

        # Converting onehot categorical to categorical value
        sampled_actor_action = actor_action[..., :].nonzero()[:, 0].cuda()
        # To choose the other character's strength, defense and attack based on the character and its type
        reactor_strength = pyro.sample("reactor_strength", dist.Categorical(self.cpts["strength"][rct_idx, rct_typ_idx])).cuda()
        reactor_defense = pyro.sample("reactor_defense", dist.Categorical(self.cpts["defense"][rct_idx, rct_typ_idx])).cuda()
        reactor_attack = pyro.sample("reactor_attack", dist.Categorical(self.cpts["attack"][rct_idx, rct_typ_idx])).cuda()

        # To choose the character's (reactor, who reacts to the actor's action in a duel) reaction based on its own strength, defense , attack and the other character's action.
        reactor_reaction = pyro.sample("reactor_reaction", dist.OneHotCategorical(self.cpts["reaction"][reactor_strength, reactor_defense, reactor_attack, sampled_actor_action])).cuda()

        #Modiying actor/reactor type tensor sizes to match the original num_labels.

        #actor_type = modify_type_tensor(actor_type, act_idx)
        #reactor_type = modify_type_tensor(reactor_type, rct_idx)

        ys = torch.cat([actor.unsqueeze(0), actor_type, actor_action, reactor.unsqueeze(0), reactor_type, reactor_reaction], dim=-1).cuda()

        z_loc = torch.zeros(1,self.z_dim,dtype=torch.float32).cuda()
        z_scale = torch.ones(1, self.z_dim, dtype=torch.float32).cuda()
        z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
        
        loc_img = self.decoder.forward(z,ys)
                # score against actual images

                #print(f"actor is {actor},reactor is {reactor}, actor_type is {actor_type}, reactor_type is {reactor_type},actor_strength is {actor_strength}, actor_defense is {actor_defense},actor_attack is {actor_attack}, actor_action is {actor_action}, sampled_actor_action is {sampled_actor_action}, reactor_strength is {reactor_strength}, reactor_attack is {reactor_attack}, reactor_defense is {reactor_defense},reactor_reaction is {reactor_reaction}, ys is {ys}")
                # return the loc so we can visualize it later
        model_attrs = {
            "actor": actor,
            "actor_type": actor_type,
            "action": actor_action,
            "reactor": reactor,
            "reactor_type": reactor_type,
            "reaction": reactor_reaction,
            "ys": ys
        }
        return loc_img, model_attrs

    def inference_guide(self, x):
      pass


    # define a helper function for reconstructing images
     # define a helper function for reconstructing images
    def reconstruct_img(self, x, y):
        # encode image x
        z_loc, z_scale = self.encoder(x,y)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z, y)
        return loc_img


