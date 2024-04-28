import torch
from torch import nn
from networks.attention import sequence_mask


class MaskedCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, Y_valid_lens):
        weights = torch.ones_like(pred)
        weights = sequence_mask(weights, Y_valid_lens)
        self.reduction = 'none'
        unweighted_loss = super(MaskedCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss*weights).mean(dim=1)
        return weighted_loss


def grad_clipping(net, theta):
    params = [p for p in net.parameters()]
    norm = torch.sqrt(sum(torch.sum((p.grad**2))for p in params))
    if norm>theta:
        for p in params:
            p.grad[:] *= theta/norm


def train(net, data_iter, num_epochs, lr, tgt_vocab, device):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedCELoss()

    for epoch in range(num_epochs):
        num_tokens, total_loss = 0, 0
        for batch in data_iter:
            optimizer.zero_grad()

            X, X_valid_lens, Y, Y_valid_lens = batch
            bos = torch.tensor([tgt_vocab['bos']]*Y.shape[0], device=device).reshape(-1, 1)
            dec_inputs = torch.cat(bos, Y[:, :-1], 1)

            Y_hat, _ = net(X, dec_inputs, X_valid_lens)

            l = loss(Y_hat, Y, Y_valid_lens)
            l.sum().backward()
            grad_clipping(net, 1)
            optimizer.step()
            with torch.no_grad():
                total_loss += l.sum()
                num_tokens += Y_valid_lens
        if (epoch+1)%10 == 0:
            print(total_loss/num_tokens)
    torch.save(net.state_dict(), '/model/final.pth')




            


