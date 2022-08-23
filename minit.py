import torch
import torch.nn.functional as F
import random
from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., drophead=0):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.attention = nn.Identity()

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.drophead = drophead

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        something = self.attention(attn.clone().detach())

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        ## drop the results of heads based on the drophead rate during training.
        ## Same drop result for every example in batch.
        #if self.training:
        #    #masks = np.reshape(np.array([round(random.random()) for i in range(h)]), (1,h,1,1))
        #    #masks = x.new_zeros((1,h,1,1))
        #    masks = torch.bernoulli(torch.tensor([1-self.drophead for i in range(h)], device=x.device).float()).view(1,h,1,1)
        #    out *= masks 
        #    if masks.sum() > 0:
        #        out *= h/masks.sum()

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., drophead=0, layerdrop=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, drophead=drophead))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
        self.layerdrop = layerdrop
        self.drophead = drophead

    def forward(self, x, mask = None):
        drop_locations = [] # these are the indices to drop.
        if self.drophead != 0:
            if self.training:
                for i in range(len(self.layers)):
                    if random.random() < self.drophead:
                        drop_locations.append(i)
            else:
                # For evaluation, use the "drop every other layer strategy" outlined in 
                # https://arxiv.org/pdf/1909.11556.pdf
                cur = 1
                while cur < len(self.layers):
                    drop_locations.append(cur)
                    cur += 1/self.drophead

        # Different sort of pruning
        for i, (attn, ff) in enumerate(self.layers):
            if i in drop_locations:
                continue
            x = attn(x, mask = mask)
            x = ff(x)
        '''
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        '''
        return x

class ViT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., drophead=0, layerdrop=0):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert layerdrop==0 or int(1/layerdrop) == 1/layerdrop, '1/layerdrop needs to be an integer'
        num_patches = (image_size // patch_size) ** 3
        patch_dim = channels * patch_size ** 3
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (l p3) -> b (h w l) (p1 p2 p3 c)', p1 = patch_size, p2 = patch_size, p3 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # 1, 4**3 + 1, 512
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, drophead=drophead, layerdrop=layerdrop)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # block_embedding is the positional embedding for the block.
    # This should have shape (b, n+1, d) so we can add to x.
    def forward(self, img, mask = None, block_embedding=None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x += block_embedding # Adding the block embedding.
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        #print(x.shape)
        #return x.view(b, -1) # x should be shape (b, (#patches +1) * dim) now. TODO for some reason this didn't work as well.
        #print(x.shape)
        return self.mlp_head(x)

class MINiT(nn.Module):
    # All the small parameters are going to be fed into ViT for the small blocks.
    def __init__(self, *, block_size, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., drophead=0, layerdrop=0, amp_enabled=False, **kwargs):
            super().__init__()
            self.image_size = image_size
            self.block_size = block_size
            self.block_count = self.image_size//self.block_size # block count per side (block_count**3 total blocks)
            self.channels = channels
            self.num_classes = num_classes
            self.patch_size = patch_size
            self.dim = dim
            self.vit = ViT(
                    image_size = block_size,
                    patch_size = patch_size, 
                    num_classes = num_classes,
                    dim = dim, 
                    depth = depth, 
                    heads = heads, 
                    mlp_dim = mlp_dim, 
                    pool = pool, 
                    channels=channels, 
                    dim_head=dim_head, 
                    dropout=dropout, 
                    emb_dropout=emb_dropout,
                    drophead=drophead,
                    layerdrop=layerdrop)
            self.linear = nn.Linear(self.block_count**3 * self.num_classes, self.num_classes)
            self.block_embeddings = nn.Parameter(torch.randn(self.block_count**3, (block_size//patch_size)**3+1, dim)) # 4**3, 4**3 + 1, 512
            self.amp_enabled = amp_enabled

    def forward(self, img):
        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            b = img.shape[0]
            p = self.block_size # this is side length
            block_count = self.block_count
            x = rearrange(img, 'b c (h p1) (w p2) (l p3) -> (b h w l) c p1 p2 p3', p1 = p, p2 = p, p3=p)
            results = self.vit(x, block_embedding=self.block_embeddings.repeat(b, 1, 1)).float()
            results = rearrange(results, '(b h w l) n ->  b (h w l n)', h = block_count, w = block_count, l=block_count, n = self.num_classes)
            logits = self.linear(results)

        return logits

if __name__ == '__main__':
    net = MINiT(
        block_size = 16,
        image_size = 64,
        patch_size = 4,
        num_classes = 2,
        channels = 1,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 309
    )

    test = torch.ones(2, 1, 64, 64, 64)
    preds = net(test)
