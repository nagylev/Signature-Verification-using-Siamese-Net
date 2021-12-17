import torch


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, alfa, beta, margin=1):
        super().__init__()
        self.alfa = alfa
        self.beta = beta
        self.margin = margin

    # s1, s2 each a sign y is the label of the pair
    # y=1, genuine,genuine
    # y=0, genuine,forged
    def forward(self, s1, s2, y):
        distance = torch.pairwise_distance(s1, s2)

        p1 = self.alfa * (1 - y) * distance ** 2
        pmax = torch.max(torch.zeros_like(distance), self.margin - distance)
        p2 = self.beta * y * pmax ** 2

        calc_loss = p1 + p2

        mean_loss = torch.mean(calc_loss, dtype=torch.float)

        return mean_loss


# calculating accuracy with the given distances
def accuracy(distances, y, step=0.01):
    minimum_distance = min(distances)
    maximum_distance = max(distances)
    max_acc = 0
    same_id = (y == 1)

    for distance in torch.arange(minimum_distance, maximum_distance + step, step):
        true_positive = (distances <= distance) & same_id
        true_positive_rate = true_positive.sum().float() / same_id.sum().float()
        true_negative = (distances > distance) & ~same_id
        true_negative_rate = true_negative.sum().float() / (~same_id).sum().float()

        acc = 0.5 * (true_negative_rate + true_positive_rate)
        max_acc = max(max_acc, acc)
    return max_acc
