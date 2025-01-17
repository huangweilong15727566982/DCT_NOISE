import numpy as np
import torch
from torch.autograd import Variable
from util import kl_divergence_loss
from dct import DCTLayer,IDCTLayer, NoiseTransform,create_perturbation_mask,add_perturbation_with_mask,dct_on_batches,idct_on_batches
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class PerturbationTool():
    def __init__(self, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725,c=None):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        if c!=None:
            self.channel=int(c)
            self.dct_on_batch=NoiseTransform().dct_on_batches
            self.idct_on_batch=NoiseTransform().idct_on_batches
            self.add_noise=NoiseTransform().add_noise
        np.random.seed(seed)

    def random_noise(self, noise_shape=[10, 3, 32, 32]): 
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device) 
        return random_noise  

    def min_min_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True) 
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps): 
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            perturb_img.retain_grad()
            loss.backward()
            eta = self.step_size * perturb_img.grad.data.sign() * (-1) 
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True) 
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon) #
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
           
        return perturb_img, eta
    def min_max_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            loss.backward()

            eta = self.step_size * perturb_img.grad.data.sign()
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon) 
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta
    def min_max_attack2(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(perturb_img)
               
                new_labels=torch.where(labels<9,labels+1,0)
                loss=criterion(logits,new_labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            loss.backward()
            eta = self.step_size * perturb_img.grad.data.sign() 
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon) 
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta
    def _patch_noise_extend_to_img(self, noise, image_size=[3, 32, 32], patch_location='center'):
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

        if patch_location == 'center' or (h == w == x_len == y_len):
            x = h // 2
            y = w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, w - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
        else:
            raise('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, h)
        x2 = np.clip(x + x_len // 2, 0, h)
        y1 = np.clip(y - y_len // 2, 0, w)
        y2 = np.clip(y + y_len // 2, 0, w)
        if type(noise) is np.ndarray:
            pass
        else:
            mask[:, x1: x2, y1: y2] = noise.cpu().numpy()
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))
    def dct_attack(self, images, labels, model, optimizer, criterion, channelnum=20, random_noise=None,
                       sample_wise=False):
        mask = create_perturbation_mask(*images.shape, num_elements=self.channel)
        dct_on_batch = DCTLayer().requires_grad_(True).cuda()
        reversed_dct_on_batch = IDCTLayer().requires_grad_(True).cuda()

        dct_images = torch.tensor(dct_on_batch(images)).float().to(device).requires_grad_(
            True)  

        if random_noise is None:  
            shape = list(dct_images.shape)
            shape[1] = channelnum
            shape = tuple(shape)
            random_noise = torch.FloatTensor(*shape).uniform_(-self.epsilon, self.epsilon).to(
                device)  

     
        perturb_on_dct = torch.tensor(dct_on_batch(random_noise)).float().to(device).requires_grad_(True) 
        perturb_imgs = add_perturbation_with_mask(dct_images, perturb_on_dct, mask).requires_grad_(True)
        for _ in range(self.num_steps):  
            opt = torch.optim.SGD([perturb_on_dct], lr=1e-3) 
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):  
                if hasattr(model, 'classify'):
                    model.classify = True
                logits = model(reversed_dct_on_batch(perturb_imgs))
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, reversed_dct_on_batch(perturb_imgs), labels, optimizer)
              
            perturb_on_dct.retain_grad()  
            loss.backward()  
            eta = self.step_size * perturb_on_dct.grad.data.sign() * (-1) 
            perturb_on_dct = perturb_on_dct.data.add_(eta).requires_grad_(True)
            perturb_on_dct = torch.clamp(perturb_on_dct, -self.epsilon, self.epsilon).requires_grad_(True) 
            perturb_on_dct = torch.tensor(perturb_on_dct).to(device).requires_grad_(True)
            perturb_imgs = add_perturbation_with_mask(dct_images, perturb_on_dct, mask).requires_grad_(True)
        return perturb_imgs, reversed_dct_on_batch(perturb_on_dct)  
    def dct_attack2(self, images, labels, model, optimizer, criterion, channelnum=20, random_noise=None,
                        sample_wise=False):
            mask = create_perturbation_mask(*images.shape, num_elements=self.channel)
            dct_on_batch = DCTLayer().requires_grad_(True).cuda()
            reversed_dct_on_batch = IDCTLayer().requires_grad_(True).cuda()
            dct_images = torch.tensor(dct_on_batch(images)).float().to(device).requires_grad_(
                True)  

            if random_noise is None: 
                shape = list(dct_images.shape)
                shape[1] = channelnum
                shape = tuple(shape)
                random_noise = torch.FloatTensor(*shape).uniform_(-self.epsilon, self.epsilon).to(
                    device)  

           
            perturb_on_dct = torch.tensor(dct_on_batch(random_noise)).float().to(device).requires_grad_(True) 
            perturb_imgs = add_perturbation_with_mask(dct_images, perturb_on_dct, mask).requires_grad_(True)
            for _ in range(self.num_steps):  
                opt = torch.optim.SGD([perturb_on_dct], lr=1e-3)  
                opt.zero_grad()
                model.zero_grad()
                if isinstance(criterion, torch.nn.CrossEntropyLoss): 
                    if hasattr(model, 'classify'):
                        model.classify = True
                    logits1 = model(reversed_dct_on_batch(perturb_imgs))
                    loss1 = criterion(logits1, labels)
                    logits2 = model(images)
                    loss2 = kl_divergence_loss(logits1,logits2)
                    loss=loss1-loss2
                else:
                    logits, loss = criterion(model, reversed_dct_on_batch(perturb_imgs), labels, optimizer)
                perturb_on_dct.retain_grad()  
                loss.backward() 
                eta = self.step_size * perturb_on_dct.grad.data.sign() * (-1) 
                perturb_on_dct = perturb_on_dct.data.add_(eta).requires_grad_(True)
                perturb_on_dct = torch.clamp(perturb_on_dct, -self.epsilon, self.epsilon).requires_grad_(True) 
                perturb_on_dct = torch.tensor(perturb_on_dct).to(device).requires_grad_(True)
                perturb_imgs = add_perturbation_with_mask(dct_images, perturb_on_dct, mask).requires_grad_(True)
            return perturb_imgs, reversed_dct_on_batch(perturb_on_dct) 
