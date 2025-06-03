import torch
from torch import nn, optim
import torch.nn.functional as F
from pytorch_lightning import LightningModule # type: ignore


class EncoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, downsample=True, input_type='1d', residual=False):
        super(EncoderModule, self).__init__()
        self.input_type = input_type
        self.conv, self.pool, self.batch_norm = None, None, None
        self.conv_res = None
        self.residual = residual
        if input_type == '1d':
            self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding='same')
            if residual: self.conv_res = nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding='same')
            if downsample: self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.batch_norm = nn.BatchNorm1d(output_channels)
        else:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding='same')
            if residual: self.conv_res = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding='same')
            if downsample: self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.batch_norm = nn.BatchNorm2d(output_channels)
            
    def forward(self, x):
        
        if self.residual: id = self.conv_res(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.residual: x = x + id
        x = F.relu(x)
        if self.pool: x = self.pool(x)
        return x
    

class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, upsample=True, input_type='1d', residual=False, apply_activation=True):
        super(DecoderModule, self).__init__()
        self.input_type = input_type
        self.conv, self.upsample, self.batch_norm = None, None, None
        self.conv_res = None
        self.residual = residual
        self.apply_activation = apply_activation
        if input_type == '1d':
            self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding='same')
            if residual: self.conv_res = nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding='same')
            self.batch_norm = nn.BatchNorm1d(output_channels)
        else:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding='same')
            if residual: self.conv_res = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding='same')
            self.batch_norm = nn.BatchNorm2d(output_channels)
        if upsample: self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        if self.residual: id = self.conv_res(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.residual: x = x + id
        if self.apply_activation: x = F.relu(x)
        if self.upsample: x = self.upsample(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_channels, embedding_size, input_shape, n_modules=4, max_conv_channels=256, input_type='1d', verbose=False):
        super(Encoder, self).__init__()
        self.verbose = verbose
        self.input_type = input_type
        self.input_shape = input_shape
        
        # Convolutional layers        
        self.encoder_modules = []
        output_channels = 128
        n_samples_per_channel=input_shape[-1]
        
        # downsample = False if input_type != '1d' else True if n_samples_per_channel % 2 == 0 else False
        downsample = False if input_type != '1d' else True if n_samples_per_channel % 2 == 0 else False
        if downsample: n_samples_per_channel = n_samples_per_channel // 2
        self.encoder_modules.append( EncoderModule(input_channels, output_channels, input_type=input_type, downsample=downsample) )
        
        
        for i in range(1, n_modules//2):
            residual = i%2 != 0 # residual connection every 2 layers
            input_channels = output_channels
            output_channels = min(2*input_channels, max_conv_channels)
            downsample = False if input_type != '1d' else True if n_samples_per_channel % 2 == 0 else False
            if downsample: n_samples_per_channel = n_samples_per_channel // 2
            self.encoder_modules.append(EncoderModule(input_channels, output_channels, input_type=input_type, downsample=downsample, residual=residual))
        
        input_channels = output_channels
        for i in range(n_modules//2, n_modules):
            residual = i%2 != 0 # residual connection every 2 layers
            output_channels = int(input_channels/2)
            downsample = False if input_type != '1d' else True if n_samples_per_channel % 2 == 0 else False
            if downsample: n_samples_per_channel = n_samples_per_channel // 2
            self.encoder_modules.append( EncoderModule(input_channels, output_channels, input_type=input_type, downsample=downsample, residual=residual) )
            input_channels = output_channels
        
        self.encoder_modules = nn.ModuleList(self.encoder_modules)

        # Linear layers
        self.last_output_channels = output_channels 
        self.last_n_samples_per_channel = n_samples_per_channel
        
        fc_intermediate_dim = embedding_size // 2
        print(self.last_output_channels, self.last_n_samples_per_channel,self.last_output_channels*self.last_n_samples_per_channel)
        fc1_input_dim = self.last_output_channels*self.last_n_samples_per_channel if input_type == '1d' else self.input_shape[-2]*self.input_shape[-1]
        # fc1_input_dim = self.last_output_channels*self.last_n_samples_per_channel if input_type == '1d' else output_channels*self.input_shape[-2]*self.input_shape[-1]
        self.fc1 = nn.Linear(fc1_input_dim, fc_intermediate_dim)
        self.fc2 = nn.Linear(fc_intermediate_dim, embedding_size)
        
        
    def forward(self, x):
        if self.verbose: print('##### Enc start: ', x.shape)

        # apply convolutional layers
        for i, encoder_module in enumerate(self.encoder_modules):
            x = encoder_module(x)
            if self.verbose: print(f'after module{i}: \t', x.shape)
        
        # TODO think about better ways to reduce dimensions
        if self.input_type != '1d':
            x = x.sum(dim=(1))
        x = x.flatten(1)            
        
        # apply linear layers
        if self.verbose: print('after flattening: \t', x.shape)
        x = F.relu(self.fc1(x))
        if self.verbose: print('after fc1: \t', x.shape)
        x = F.relu(self.fc2(x))
        if self.verbose: print('after fc2: \t', x.shape)
        
        if self.verbose: print('##### Enc end')
        return x
    

class Decoder(nn.Module):
    def __init__(self, enc_layers, input_shape, input_type='1d', verbose=False):
        super(Decoder, self).__init__()
        self.verbose = verbose
        self.input_type = input_type
        self.input_shape = input_shape
        
        self.linear_layers = []
        self.decoder_modules = []
        
        enc_layers.reverse()
        for layer in enc_layers:
            if layer[0] == 'fc':
                self.linear_layers.append(nn.Linear(layer[1][0], layer[1][1]))
            elif layer[0] == 'module_list':
                modules = layer[1]
                modules.reverse()
                for i, sub_layer in enumerate(modules):
                    apply_activation = True if i < len(modules)-1 else False
                    self.decoder_modules.append(DecoderModule(sub_layer[1][0], sub_layer[1][1], input_type=input_type, upsample=sub_layer[2], apply_activation=apply_activation))
        
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.decoder_modules = nn.ModuleList(self.decoder_modules)
        
        
    def forward(self, x):
        if self.verbose: print('##### Dec start: ', x.shape)
        
        # apply linear layers
        for i, linear_layer in enumerate(self.linear_layers):
            x = linear_layer(x)
            if self.verbose: print(f'after fc{i}: \t', x.shape)
        
        in_channels = self.decoder_modules[0].conv.in_channels
        if self.input_type == '1d': 
            seq_len = x.size(-1) // in_channels
            x = x.view(x.size(0), in_channels, seq_len) 
        else: 
            # x = x.view(x.size(0), in_channels, self.input_shape[-2], self.input_shape[-1]) 
            # TODO think about better ways to reduce dimensions:
            x = x.view(x.size(0), 1, self.input_shape[-2], self.input_shape[-1]) 
            x = x.repeat(1, in_channels, 1, 1)
        
        if self.verbose: print('after unflattening: \t', x.shape)
        
        # apply convolutional layers
        for i, decoder_module in enumerate(self.decoder_modules):
            x = decoder_module(x)
            if self.verbose: print(f'after module{i}: \t', x.shape)
        
        if self.verbose: print('##### Dec end')
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_channels, embedding_size, input_shape, n_modules=8, input_type='1d'):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_channels, embedding_size, input_shape, input_type=input_type, n_modules=n_modules)
        
        enc_layers = []
        for name, layer in self.encoder.named_children():
            if isinstance(layer, nn.Linear): 
                enc_layers.append( ('fc', layer.weight.shape) )
            elif isinstance(layer, nn.ModuleList): 
                module_list = []
                for sub_name, sub_layer in enumerate(layer): 
                    module_list.append( ('conv', sub_layer.conv.weight.shape, sub_layer.pool!=None) )
                enc_layers.append( ('module_list', module_list) )
        
        self.decoder = Decoder(enc_layers, input_shape, input_type=input_type)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def forward_all_outputs(self, x):
        embedding = self.encoder(x)
        x = self.decoder(embedding)
        return x, embedding
    

class AutoEncoderPL(LightningModule):
    def __init__(self, input_channels, embedding_size, input_shape, input_type='1d', n_modules=8, lr=1e-3):
        super().__init__()
        self.input_type = input_type
        self.model = AutoEncoder(input_channels, embedding_size, input_shape, input_type=input_type, n_modules=n_modules)
        self.lr = lr
        
    def loss(self, x, x_hat):
        # return torch.sqrt(nn.functional.mse_loss(x_hat, x))
        return nn.functional.mse_loss(x_hat, x, reduction='mean')
    
    def training_step(self, batch, batch_idx):
        x = None
        if self.input_type == '1d': x = batch['data'].float()
        else: x = batch['spectrograms'].float()
        
        # # adding gaussian noise to make embeddings more robust
        # noise = torch.randn_like(x) * 0.01 # + mean
        # x = x + noise
        # # TODO try other types of noise (Masking, in frequency and/or time)
        
        out = self.model(x)
        loss = self.loss(out, x)
        self.log('train_loss', loss, on_epoch=True, sync_dist=True, batch_size=x.size(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = None
        if self.input_type == '1d': x = batch['data'].float()
        else: x = batch['spectrograms'].float()
        
        # # adding gaussian noise to make embeddings more robust
        # noise = torch.randn_like(x) * 0.01 # + mean
        # x = x + noise
        # # TODO try other types of noise (Masking, in frequency and/or time)
        
        out = self.model(x)
        loss = self.loss(out, x)
        self.log('val_loss', loss, on_epoch=True, sync_dist=True, batch_size=x.size(0))
        return loss
    
    def test_step(self, batch, batch_idx):
        x = None
        if self.input_type == '1d': x = batch['data'].float()
        else: x = batch['spectrograms'].float()
        
        # # adding gaussian noise to make embeddings more robust
        # noise = torch.randn_like(x) * 0.01 # + mean
        # x = x + noise
        # # TODO try other types of noise (Masking, in frequency and/or time)
        
        out = self.model(x)
        loss = self.loss(out, x)
        self.log('test_loss', loss, on_epoch=True, sync_dist=True, batch_size=x.size(0))
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, x):
        return self.model(x.float())
    
    def set_verbosity(self, verbose):
        self.model.encoder.verbose = verbose
        self.model.decoder.verbose = verbose