
import torch
import torchvision
from Network import Network

def test(net, test_loader, epoch):

    net.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    samples = 0
    with torch.no_grad():

        for data, target in test_loader:
            t = target.numpy()
            output = net(data)
            #test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            samples += len(data)

        per = 100 * float(correct)/samples
        print('epoch {}, Accuracy: {}/{} - {})'.format(epoch, correct, samples, per))
        return per