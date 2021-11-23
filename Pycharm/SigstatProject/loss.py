import torch


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, alfa, beta, margin=1):
        super().__init__()
        self.alfa = alfa
        self.beta = beta
        self.margin = margin

    def forward(self, s1, s2, y):
        distance = torch.cdist(s1, s2)  # .pairwise_distance(x1, x2)
        l = self.alfa * (1 - y) * distance ** 2 + self.beta * y * (max(0, self.margin - distance)) ** 2
        mean_loss = torch.mean(l, dtype=torch.float)

        return mean_loss
