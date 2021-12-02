import torch



class ContrastiveLoss(torch.nn.Module):

    def __init__(self, alfa, beta, margin=1):
        super().__init__()
        self.alfa = alfa
        self.beta = beta
        self.margin = margin

    #s1, s2 egy-egy alairas
    def forward(self, s1, s2, y):
        distance = torch.pairwise_distance(s1, s2)

        print(distance)
        p1 = self.alfa * (1 - y) * distance ** 2
        pmax = torch.max(torch.zeros_like(distance), self.margin - distance)
        p2 = self.beta * y * pmax ** 2

        calc_loss = p1 + p2

        mean_loss = torch.mean(calc_loss, dtype=torch.float)

        return mean_loss
