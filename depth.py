from mtcnn.nets import *


def depth(net):
    return sum(p.numel() for p in net.parameters())


print(depth(PNet()))  # 6632
print(depth(RNet()))  # 100178
print(depth(ONet()))  # 389040
