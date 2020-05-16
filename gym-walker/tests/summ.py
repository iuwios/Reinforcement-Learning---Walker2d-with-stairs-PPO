from torch.utils.tensorboard import SummaryWriter
import numpy as np
writer = SummaryWriter()
for i in range(10):
    x = np.random.random(1000)
    writer.add_histogram('distribution centers', x + i, i)
    add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
writer.close()