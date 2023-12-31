# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/036_models.InceptionTimePlus.ipynb.

# %% auto 0
__all__ = ['InceptionModulePlus', 'InceptionBlockPlus', 'InceptionTimePlus']

# %% ../../nbs/036_models.InceptionTimePlus.ipynb 4
from imports_2 import *
from collections import OrderedDict
from fastai.layers import *
from utils_3 import *
from layers_1 import *
from utils_1 import *

# %% ../../nbs/036_models.InceptionTimePlus.ipynb 5
# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@timeseriesAI.co modified from:

# Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., ... & Petitjean, F. (2019). 
# InceptionTime: Finding AlexNet for Time Series Classification. arXiv preprint arXiv:1909.04939.
# Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime

    
class InceptionModulePlus(Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True, padding='same', coord=False, separable=False, dilation=1, stride=1, conv_dropout=0., sa=False, se=None,
                 norm='Batch', zero_norm=False, bn_1st=True, act=nn.ReLU, act_kwargs={}):
        
        if not (is_listy(ks) and len(ks) == 3):
            if isinstance(ks, Integral): ks = [ks // (2**i) for i in range(3)]
            ks = [ksi if ksi % 2 != 0 else ksi - 1 for ksi in ks]  # ensure odd ks for padding='same'

        bottleneck = False if ni == nf else bottleneck
        self.bottleneck = Conv(ni, nf, 1, coord=coord, bias=False) if bottleneck else noop # 
        self.convs = nn.ModuleList()
        for i in range(len(ks)): self.convs.append(Conv(nf if bottleneck else ni, nf, ks[i], padding=padding, coord=coord, separable=separable,
                                                         dilation=dilation**i, stride=stride, bias=False))
        self.mp_conv = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), Conv(ni, nf, 1, coord=coord, bias=False)])
        self.concat = Concat()
        self.norm = Norm(nf * 4, norm=norm, zero_norm=zero_norm)
        self.conv_dropout = nn.Dropout(conv_dropout) if conv_dropout else noop
        self.sa = SimpleSelfAttention(nf * 4) if sa else noop
        self.act = act(**act_kwargs) if act else noop
        self.se = nn.Sequential(SqueezeExciteBlock(nf * 4, reduction=se), BN1d(nf * 4)) if se else noop
        
        self._init_cnn(self)
    
    def _init_cnn(self, m):
        if getattr(self, 'bias', None) is not None: nn.init.constant_(self.bias, 0)
        if isinstance(self, (nn.Conv1d,nn.Conv2d,nn.Conv3d,nn.Linear)): nn.init.kaiming_normal_(self.weight)
        for l in m.children(): self._init_cnn(l)

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(x)
        x = self.concat([l(x) for l in self.convs] + [self.mp_conv(input_tensor)])
        x = self.norm(x)
        x = self.conv_dropout(x)
        x = self.sa(x)
        x = self.act(x)
        x = self.se(x)
        return x


@delegates(InceptionModulePlus.__init__)
class InceptionBlockPlus(Module):
    def __init__(self, ni, nf, residual=True, depth=6, coord=False, norm='Batch', zero_norm=False, act=nn.ReLU, act_kwargs={}, sa=False, se=None, 
                 stoch_depth=1., **kwargs):
        self.residual, self.depth = residual, depth
        self.inception, self.shortcut, self.act = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            self.inception.append(InceptionModulePlus(ni if d == 0 else nf * 4, nf, coord=coord, norm=norm, 
                                                      zero_norm=zero_norm if d % 3 == 2 else False,
                                                      act=act if d % 3 != 2 else None, act_kwargs=act_kwargs, 
                                                      sa=sa if d % 3 == 2 else False,
                                                      se=se if d % 3 != 2 else None,
                                                      **kwargs))
            if self.residual and d % 3 == 2:
                n_in, n_out = ni if d == 2 else nf * 4, nf * 4
                self.shortcut.append(Norm(n_in, norm=norm) if n_in == n_out else ConvBlock(n_in, n_out, 1, coord=coord, bias=False, norm=norm, act=None))
                self.act.append(act(**act_kwargs))
        self.add = Add()
        if stoch_depth != 0: keep_prob = np.linspace(1, stoch_depth, depth)
        else: keep_prob = np.array([1] * depth)
        self.keep_prob = keep_prob

    def forward(self, x):
        res = x
        for i in range(self.depth):
            if self.keep_prob[i] > random.random() or not self.training:
                x = self.inception[i](x)
            if self.residual and i % 3 == 2: 
                res = x = self.act[i//3](self.add(x, self.shortcut[i//3](res)))
        return x

# %% ../../nbs/036_models.InceptionTimePlus.ipynb 6
@delegates(InceptionModulePlus.__init__)
class InceptionTimePlus(nn.Sequential):
    def __init__(self, c_in, c_out, seq_len=None, nf=32, nb_filters=None,
                 flatten=False, concat_pool=False, fc_dropout=0., bn=False, y_range=None, custom_head=None, **kwargs):
        
        if nb_filters is not None: nf = nb_filters
        else: nf = ifnone(nf, nb_filters) # for compatibility
        backbone = InceptionBlockPlus(c_in, nf, **kwargs)
        
        #head
        self.head_nf = nf * 4
        self.c_out = c_out
        self.seq_len = seq_len
        if custom_head is not None: 
            if isinstance(custom_head, nn.Module): head = custom_head
            else: head = custom_head(self.head_nf, c_out, seq_len)
        else: head = self.create_head(self.head_nf, c_out, seq_len, flatten=flatten, concat_pool=concat_pool, 
                                      fc_dropout=fc_dropout, bn=bn, y_range=y_range)
            
        layers = OrderedDict([('backbone', nn.Sequential(backbone)), ('head', nn.Sequential(head))])
        super().__init__(layers)
        
    def create_head(self, nf, c_out, seq_len, flatten=False, concat_pool=False, fc_dropout=0., bn=False, y_range=None):
        #print(f'flatten: {flatten}')
        if flatten: 
            nf *= seq_len
            layers = [Flatten()]
        else: 
            if concat_pool: nf *= 2
            layers = [GACP1d(1) if concat_pool else GAP1d(1)]
        layers += [LinBnDrop(nf, c_out, bn=bn, p=fc_dropout)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)

