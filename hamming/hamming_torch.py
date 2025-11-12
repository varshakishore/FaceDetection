import torch


class Hamming74(object):
    def __init__(self, device="cpu"):
        self.G = torch.tensor([[1, 1, 0, 1],
                               [1, 0, 1, 1],
                               [1, 0, 0, 0],
                               [0, 1, 1, 1],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]).permute(1, 0).to(device).float()

        self.H = torch.tensor([[1, 0, 1, 0, 1, 0, 1],
                               [0, 1, 1, 0, 0, 1, 1],
                               [0, 0, 0, 1, 1, 1, 1]]).permute(1, 0).to(device).float()

        self.H2 = torch.tensor([1, 2, 4]).to(device).float()

    def encode(self, x):
        # x: torch.tensor, (kx4,)
        # output: torch.tensor, (kx7,)
        return torch.matmul(x.view(-1, 4).float(), self.G).view(-1).long() % 2

    def decode(self, x):
        # x: torch.tensor, (kx7,)
        # output: torch.tensor, (kx4,)
        x = x.reshape(-1, 7)
        z = torch.matmul(torch.matmul(x.float(), self.H) % 2, self.H2).long()
        xx = torch.tensor([i for i in range(z.shape[0]) if z[i] > 0]).long()
        yy = z[xx] - 1
        x[xx, yy] = 1 - x[xx, yy]
        return (x[:, [2, 4, 5, 6]]).view(-1) # equivalent to return (x @ R).reshape(-1)


if __name__ == "__main__":
    device = "cuda" # "cpu"
    hamming = Hamming74(device)
    x = torch.tensor([1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]).to(device)
    print(x)
    y = hamming.encode(x)
    print(y)
    x0 = x.clone()
    x[[1, 6, 11, 12]] = 1-x[[1, 6, 11, 12]]
    z = hamming.decode(y)
    print(z, torch.sum(z-x0))

