import torch

def cosine_similarity(A, B):
    """
    Cosine_similarity between two matrices
    :param A: 2d tensor
    :param B: 2d tensor
    :return:
    """
    return torch.mm(
        torch.nn.functional.normalize(A),
        torch.nn.functional.normalize(B).T
    )
