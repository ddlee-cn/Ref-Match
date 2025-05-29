import os
import numpy as np

from PIL import Image
from models import vgg19_model
from refmatch import refmatch
from util import util

from options.options import Options
opt = Options().parse()

# load VGG
vgg19 = vgg19_model.define_Vgg19(opt)


# read input
name, ref_name = opt.name, opt.ref_name
datarootA = os.path.join(opt.data_root, name+".png")
datarootB = os.path.join(opt.data_root, ref_name+".png")
datarootA_in = os.path.join(opt.in_root, name+"_{}.png".format(opt.mask_type))
mask_path = os.path.join(opt.data_root, name+"_mask_{}.png".format(opt.mask_type))

A = util.read_image(datarootA, opt.imageSize)
B = util.read_image(datarootB, opt.imageSize)
A_in =util.read_image(datarootA_in, opt.imageSize)
mask = util.read_mask(mask_path, opt.imageSize)
mask_tensor = util.make_mask(mask).cuda()

# run
rmi = refmatch.RefMatchInpainting(vgg19, opt)
rmi.run(A, B, A_in, mask)

[F_A, F_A_gt, F_B, RL_1A, F_Bm, F_Am, RL_1B_in, mapping_a_to_b_interp, mapping_a_to_b, mapping_a_to_b_gt, mapping_b_to_a] = rmi.final_list

save_dir =  os.path.join(opt.results_dir, name.replace("src", "").replace("trg", ""))
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    
# save mapping
map_concat = []
for j in [mapping_a_to_b_gt, mapping_a_to_b, mapping_a_to_b_interp, mapping_b_to_a]:
    map_concat.append(util.flow_to_color(util.mapping_to_uv(j)))
final_map = np.concatenate(map_concat, axis=1)
util.save_image(final_map, os.path.join(save_dir, "mapping.png"))

# save result
AsB = rmi.warp(F_A.size(), F_A, [3, 3], mapping_b_to_a)
A_interp = rmi.warp(AsB.size(), AsB, [3, 3], mapping_a_to_b_interp)
A_gt = rmi.warp(AsB.size(), AsB, [3, 3], mapping_a_to_b_gt)

A_mask_1 = A_gt * mask_tensor.cuda() + F_A * (1-mask_tensor.cuda())
A_mask_2 = A_interp * mask_tensor.cuda() + F_A * (1-mask_tensor.cuda())

im_concat = []
for j in [F_A_gt, AsB, A_interp, A_gt, A_mask_1, A_mask_2]:
    im_concat.append(util.tensor2im(j))
final = np.concatenate(im_concat, axis=1)

npz_file = os.path.join(opt.results_dir, name+"_{}_mapping.npz".format(opt.mask_type))
np.savez(npz_file, abi=mapping_a_to_b_interp.cpu().numpy(), ab=mapping_a_to_b.cpu().numpy(), abgt=mapping_a_to_b_gt.cpu().numpy(), ba=mapping_b_to_a.cpu().numpy())

util.save_image(final, os.path.join(save_dir, "final.png"))
util.save_image(util.tensor2im(A_mask_1), os.path.join(opt.results_dir, name+"_{}_gt.png".format(opt.mask_type)))
util.save_image(util.tensor2im(A_mask_2), os.path.join(opt.results_dir, name+"_{}.png".format(opt.mask_type)))
