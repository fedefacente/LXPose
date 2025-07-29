import torch.nn.functional as F

def sobel(x):
    a = torch.tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]], dtype=torch.float32).view((1, 1, 3, 3))

    b = torch.tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]], dtype=torch.float32).view((1, 1, 3, 3))

    # Apply reflection padding before convolution
    x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')

    G_x = F.conv2d(x_padded, a)
    G_y = F.conv2d(x_padded, b)

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

    return G
