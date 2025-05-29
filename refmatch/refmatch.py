import os
import math
import torch
import random
import numpy as np
import pyvfc
import torch.nn.functional as functional
from PIL import Image
from torch.autograd import Variable
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from . import feature_metric as FM
from util import draw_correspondence as draw
from util import util

class RefMatchInpainting():
    def __init__(self, model, opt, debug=False):
        self.Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor
        self.model = model
        self.k_per_level = 1000 # k_per_level
        self.patch_size_list = [[3,3], [5,5],[5,5],[3,3],[3,3],[3,3]]
        self.search_box_radius_list = [8, 4, 4, 4, 4, 2] #[16, 8, 4, 2, 2] # [64,32,16,4,2]
        self.pad_mode = 'reflect'
        self.L_final = 0
        self.L_start = 5
        self.tau = opt.tau
        self.sample_ratio = 0.8 # sample_ratio in ROI region
        self.max_buddy_num = 100
        self.mask_prob = 0.5 # probability threshold to identify point as masked region
        self.vfc_ratio_bound = 0
        self.debug = debug
        self.mapping_iter = 30 # max iter for finding mapping 
        self.mapping_error_bound = 0.001 # mapping change convergence bound

    def find_mapping(self, A, B, patch_size, initial_mapping, search_box_radius, roi=None):
        assert(A.size() == B.size())
        A_to_B_map = initial_mapping.clone()
        loss_map = self.Tensor(1,1,A.size(2),A.size(3))
        mapping_distance_map = self.Tensor(1,1,A.size(2),A.size(3)).fill_(0.0)
        mapping_distance_map_old = mapping_distance_map.clone()
        [dx,dy] = [math.floor(patch_size[0]/2), math.floor(patch_size[1]/2)]
        pad_size = tuple([dy,dy,dx,dx])
        A_padded = functional.pad(A, pad_size, self.pad_mode).data
        B_padded = functional.pad(B, pad_size, self.pad_mode).data

        i_range_min, i_range_max = 0, A.size(2)
        j_range_min, j_range_max = 0, A.size(3)
        if roi:
            i_range_min, i_range_max, j_range_min, j_range_max = roi
            
        iter = 0
        mapping_distance_change = 1
        while mapping_distance_change > self.mapping_error_bound:
            iter += 1
            for i in range(i_range_min, i_range_max):
                for j in range(j_range_min, j_range_max):
                    candidate_patch_A = A_padded[:,:,(i):(i+2*dx+1),(j):(j+2*dy+1)]
                    index, distance = self.find_closest_patch_index(B_padded, candidate_patch_A, initial_mapping[0,:,i,j], search_box_radius)
                    if distance > mapping_distance_map[0, 0, i, j]:
                        mapping_distance_map[0, 0, i, j] = distance
                    A_to_B_map[:,:,i,j] = self.Tensor([index[0]-dx, index[1]-dy])
            initial_mapping = A_to_B_map.clone()
            mapping_distance_change = torch.mean(torch.abs(mapping_distance_map_old - mapping_distance_map)).item()
            # if self.debug:
                # print("Find mapping iter:", iter, "mapping_distance_change:", mapping_distance_change)
            
            mapping_distance_map_old = mapping_distance_map.clone()
            if iter > self.mapping_iter:
                break

        return A_to_B_map

    def find_closest_patch_index(self, B, patch_A, inital_pixel, search_box_radius):
        """
        search inside box with template-match(conv2d implementation)
        """
        [dx, dy] = [math.floor(patch_A.size(2)/2), math.floor(patch_A.size(3)/2)]
        [search_dx, search_dy] = [search_box_radius, search_box_radius]
        up_boundary = int(inital_pixel[0]-search_dx) if inital_pixel[0]-search_dx > 0 else 0
        down_boundary = int(inital_pixel[0]+2*dx+search_dx+1) if inital_pixel[0]+2*dx+search_dx+1 < B.size(2) else B.size(2)
        left_boundary = int(inital_pixel[1]-search_dy) if inital_pixel[1]-search_dy > 0 else 0
        right_boundary = int(inital_pixel[1]+2*dy+search_dy+1) if inital_pixel[1]+2*dy+search_dy+1 < B.size(3) else B.size(3)
        search_box_B = B[:,:,up_boundary:down_boundary,left_boundary:right_boundary]
        result_B = functional.conv2d(Variable(search_box_B), Variable(patch_A.contiguous())).data
        distance = result_B
        max_j = distance.max(3)[1]
        max_i = distance.max(3)[0].max(2)[1][0][0]
        max_j = max_j[0,0,max_i]
        closest_patch_distance = distance[0,0,max_i,max_j]
        closest_patch_index = [max_i + dx + up_boundary, max_j + dy + left_boundary]

        return closest_patch_index, closest_patch_distance
    
    def warp_bi_patch(self, x, y, dx, dy, image):
        # bilinear
        x_f = max(math.floor(x), 0)
        y_f = max(math.floor(y), 0)
        x_c = min(math.ceil(x), image.size(2) - 1)
        y_c = min(math.ceil(y), image.size(3) - 1)

        b = x - x_f
        a = y - y_f
        output = (1 - a) * (1 - b) * image[:, :, x_f:(x_f+2*dx+1), y_f:(y_f+2*dy+1)] \
                 + a * (1 - b) * image[:, :, x_f:(x_f+2*dx+1), y_c:(y_c+2*dy+1)] \
                 + b * (1 - a) * image[:, :, x_c:(x_c+2*dx+1), y_f:(y_f+2*dy+1)] \
                 + a * b * image[:, :, x_c:(x_c+2*dx+1), y_c:(y_c+2*dy+1)]
        return output

    def warp(self, A_size, B, patch_size, mapping, ROI=None):
        assert(B.size() == A_size)
        [dx,dy] = [math.floor(patch_size[0]/2), math.floor(patch_size[1]/2)]
        pad_size = tuple([dy,dy,dx,dx])
        B_padded = functional.pad(B, pad_size, self.pad_mode).data
        warped_A = self.Tensor(B_padded.size()).fill_(0.0)
        counter = self.Tensor(B_padded.size()).fill_(0.0)
        for i in range(A_size[2]):
            for j in range(A_size[3]):
                map_ij = mapping[0,:,i,j]
                # warped_A[:,:,i:(i+2*dx+1),j:(j+2*dy+1)] += B_padded[:, :, int(map_ij[0]):(int(map_ij[0])+2*dx+1), int(map_ij[1]):(int(map_ij[1])+2*dy+1)]
                warped_A[:,:,i:(i+2*dx+1),j:(j+2*dy+1)] += self.warp_bi_patch(map_ij[0], map_ij[1], dx, dy, B_padded)
                counter[:,:,i:(i+2*dx+1),j:(j+2*dy+1)] += self.Tensor(B_padded.size(0),B_padded.size(1),patch_size[0],patch_size[1]).fill_(1.0)
        return warped_A[:, :, dx:(warped_A.size(2)-dx), dy:(warped_A.size(3)-dy)]/counter[:, :, dx:(warped_A.size(2)-dx), dy:(warped_A.size(3)-dy)]

    def normalize_0_to_1(self, F):
        assert(F.dim() == 4)
        max_val = F.max()
        min_val = F.min()
        if max_val != min_val:
            F_normalized = (F - min_val)/(max_val-min_val)
        else:
            F_normalized = self.Tensor(F.size()).fill_(0)

        return F_normalized

    def upsample_mapping(self, mapping, factor = 2):
        upsampler = torch.nn.Upsample(scale_factor=factor, mode='nearest')
        return upsampler(Variable(factor*mapping)).data

    def get_M(self, F, tau=0.05):
        assert(F.dim() == 4)
        F_squared_sum = F.pow(2).sum(1,keepdim=True).expand_as(F) 
        F_normalized = self.normalize_0_to_1(F_squared_sum)
        M = self.Tensor(F_normalized.size())
        M.copy_(torch.ge(F_normalized,tau))
        return M

    def identity_map(self, size):
        idnty_map = self.Tensor(size[0],2,size[2],size[3])
        idnty_map[0,0,:,:].copy_(torch.arange(0,size[2]).repeat(size[3],1).transpose(0,1))
        idnty_map[0,1,:,:].copy_(torch.arange(0,size[3]).repeat(size[2],1))
        return idnty_map

    def find_best_buddies(self, a_to_b, b_to_a, top_left_1 = [0,0], bottom_right_1 = [float('inf'), float('inf')], top_left_2 = [0,0], bottom_right_2 = [float('inf'), float('inf')]):
        assert(a_to_b.size() == b_to_a.size())
        correspondence = [[],[]]
        loss = []
        number_of_cycle_consistencies = 0
        for i in range(top_left_1[0], min(bottom_right_1[0],a_to_b.size(2))):
            for j in range(top_left_1[1], min(bottom_right_1[1],a_to_b.size(3))):
                # bidirectional mutual NN
                map_ij = a_to_b[0,:,i,j].cpu().numpy() #Should be improved (slow in cuda)
                d = util.spatial_distance(b_to_a[0,:,int(map_ij[0]),int(map_ij[1])],self.Tensor([i,j]))
                if d == 0:
                    if int(map_ij[0]) >= top_left_2[0] and int(map_ij[1]) >= top_left_2[1] and int(map_ij[0]) < bottom_right_2[0] and int(map_ij[1]) < bottom_right_2[1]:
                        correspondence[0].append([i,j])
                        correspondence[1].append([int(map_ij[0]), int(map_ij[1])])
                        number_of_cycle_consistencies += 1
        return correspondence

    def replace_refined_correspondence(self, correspondence, refined_correspondence_i, index):
        new_correspondence = correspondence
        activation = correspondence[2][index]
        new_correspondence[0].pop(index)
        new_correspondence[1].pop(index)
        new_correspondence[2].pop(index)

        for j in range(len(refined_correspondence_i[0])):
            new_correspondence[0].append(refined_correspondence_i[0][j])
            new_correspondence[1].append(refined_correspondence_i[1][j])
            new_correspondence[2].append(activation+refined_correspondence_i[2][j])

        return new_correspondence

    def remove_correspondence(self, correspondence, index):
        correspondence[0].pop(index)
        correspondence[1].pop(index)
        correspondence[2].pop(index)
        return correspondence

    def calculate_activations(self, correspondence, F_A, F_B):
        response_A = FM.stretch_tensor_0_to_1(FM.response(F_A))
        response_B = FM.stretch_tensor_0_to_1(FM.response(F_B))
        correspondence_avg_response = self.Tensor(len(correspondence[0])).fill_(0)
        response_correspondence = correspondence
        response_correspondence.append([])
        for i in range(len(correspondence[0])):
            response_A_i = response_A[0,0,correspondence[0][i][0],correspondence[0][i][1]]
            response_B_i = response_B[0,0,correspondence[1][i][0],correspondence[1][i][1]]
            correspondence_avg_response_i = (response_A_i + response_B_i)*0.5
            response_correspondence[2].append(correspondence_avg_response_i)
        return response_correspondence

    def limit_correspondence_number_per_level(self, correspondence, F_A, F_B, tau, top=5):
        correspondence_avg_response = self.Tensor(len(correspondence[0])).fill_(0)
        for i in range(len(correspondence[0])):
            correspondence_avg_response[i] = correspondence[2][i]

        top_response_correspondence = [[],[],[]]
        if len(correspondence[0]) > 0 :
            [sorted_correspondence, ind] = correspondence_avg_response.sort(dim=0, descending=True)
            for i in range(min(top,len(correspondence[0]))):
                #if self.get_M(F_A, tau=tau)[0,0,correspondence[0][ind[i]][0],correspondence[0][ind[i]][1]] == 1 and self.get_M(F_B, tau=tau)[0,0,correspondence[1][ind[i]][0],correspondence[1][ind[i]][1]] == 1:
                top_response_correspondence[0].append(correspondence[0][ind[i]])
                top_response_correspondence[1].append(correspondence[1][ind[i]])
                top_response_correspondence[2].append(sorted_correspondence[i])

        return top_response_correspondence

    def make_correspondence_unique(self, correspondence):
        unique_correspondence = correspondence
        for i in range(len(unique_correspondence[0])-1,-1,-1):
            for j in range(i-1,-1,-1):
                if self.is_same_match(unique_correspondence[0][i], unique_correspondence[0][j]):
                    unique_correspondence[0].pop(i)
                    unique_correspondence[1].pop(i)
                    unique_correspondence[2].pop(i)
                    break

        return unique_correspondence

    def is_same_match(self, corr_1, corr_2):
        if corr_1[0] == corr_2[0] and corr_1[1] == corr_2[1]:
            return True

    def response(self, F):
        response = F.pow(2).sum(1,keepdim=True)
        return response

    def top_k_in_clusters(self, correspondence, k):
        if self.debug:
            print("max_num:", k, "actual num:", len(correspondence[0]))
        if k > len(correspondence[0]):
            return correspondence

        correspondence_R_4 = []
        for i in range(len(correspondence[0])):
            correspondence_R_4.append([correspondence[0][i][0], correspondence[0][i][1], correspondence[1][i][0], correspondence[1][i][1]])

        top_cluster_correspondence = [[],[],[]]
        # print("Calculating K-means...")
        kmeans = KMeans(n_clusters=k, random_state=0).fit(correspondence_R_4)
        for i in range(k):
            max_response = 0
            max_response_index = len(correspondence[0])
            for j in range(len(correspondence[0])):
                if kmeans.labels_[j]==i and correspondence[2][j]>max_response:
                    max_response = correspondence[2][j]
                    max_response_index = j
            top_cluster_correspondence[0].append(correspondence[0][max_response_index])
            top_cluster_correspondence[1].append(correspondence[1][max_response_index])
            top_cluster_correspondence[2].append(correspondence[2][max_response_index])

        return top_cluster_correspondence
    
    def vfc_func(self, pts1, pts2, ctrlpts_id=None, ctrlpts_num=None):
        assert len(pts1)==len(pts2)
        vfc = pyvfc.VFC()
        if vfc.setData(pts1, pts2):
            if ctrlpts_id:
                vfc.setCtrlPts(ctrlpts_id)
            if ctrlpts_num:
                vfc.setCtrlPtsNum(ctrlpts_num)
            vfc.optimize()
            match_idx = vfc.obtainCorrectMatch()
            prob = vfc.obtainP()
            vecfield = vfc.obtainVecField()
            ctrlpts = vfc.obtainCtrlPts()
            return match_idx, prob, vecfield, ctrlpts
        else:
            print(pts1, pts2)
            return None, None, None, None

    def prob_to_mask(self, pts, prob, roi, shape):
        # threshold prob from VFC for mask estimation
        mask = self.Tensor(1, shape[0], shape[1]).fill_(0.0)
        for i, p in enumerate(prob):
            if not util.in_roi(roi, pts[i]):
                continue
            if p < self.mask_prob:
                mask[0, pts[i][0], pts[i][1]] = 1.0
        return mask

    def find_neural_best_buddies_roi(self, correspondence, F_A, F_B, a_to_b, b_to_a, search_box_radius, roi, L, deepest_level=False):
        # find neural best buddies for the next layer
        if deepest_level == True:
            correspondence = self.find_best_buddies(a_to_b, b_to_a)
            correspondence = self.calculate_activations(correspondence, F_A, F_B)
        else:
            removed = []
            # select buddies based on distance to the roi region
            # buddies_distance = util.chkd_roi(correspondence[0], roi)
            # if self.debug:
            #     print("buddy num == 0:", sum(np.array(buddies_distance == 0)))
            # buddy_num = min(sum(np.array(buddies_distance == 0)), len(correspondence[0]))
            # picked_buddies_idx = np.argsort(buddies_distance)[:buddy_num]
            for i in range(len(correspondence[0])-1,-1,-1):
                # only update best buddies near the ROI region (*2 to next layer), based on the layer
                if L == 0:
                    dis = util.chkd_roi([(correspondence[0][i][0], correspondence[0][i][1])], roi)
                else:
                    dis = util.chkd_roi([(correspondence[0][i][0]*2, correspondence[0][i][1]*2)], roi)                
                if dis > search_box_radius: # > (L+2) * search_box_radius:                if dis > search_box_radius: # > (L+2) * search_box_radius:
                    removed.append(correspondence[0][i])
                    correspondence = self.remove_correspondence(correspondence, i)
                    continue
                [top_left_1, bottom_right_1] = util.extract_receptive_field(correspondence[0][i][0], correspondence[0][i][1], search_box_radius, [a_to_b.size(2), a_to_b.size(3)])
                [top_left_2, bottom_right_2] = util.extract_receptive_field(correspondence[1][i][0], correspondence[1][i][1], search_box_radius, [a_to_b.size(2), a_to_b.size(3)])
                refined_correspondence_i = self.find_best_buddies(a_to_b, b_to_a, top_left_1, bottom_right_1, top_left_2, bottom_right_2)
                refined_correspondence_i = self.calculate_activations(refined_correspondence_i, F_A, F_B)
                correspondence = self.replace_refined_correspondence(correspondence, refined_correspondence_i, i)
            # print("out-of-roi correspondence removed:", removed)
        
        refined_correspondence = self.make_correspondence_unique(correspondence)
        if self.k_per_level < float('inf'):
            refined_correspondence = self.top_k_in_clusters(refined_correspondence, int(self.k_per_level/(2**(L-1))))
        
        return refined_correspondence
    
    def vfc_local_interp(self, mapping, roi, mask_cond=None, est_mask=False, patchsize=3):
        # construct candidate correspondence pairs for vector field consensus check        
        mapping_pts_left, mapping_pts_right = util.mapping_to_pairs(mapping, roi)
        
        match_mapping, prob_mapping, vecfield_mapping, ctrlpts_mapping = self.vfc_func(mapping_pts_left, mapping_pts_right)
        if match_mapping is None:
            return mapping, None
        vfc_ratio_mapping = len(match_mapping) / len(mapping_pts_left) * 1.0

        # interpolate with VFC's prediction
        mapping_interp = util.pairs_to_mapping(mapping, mapping_pts_left, vecfield_mapping, roi, mask_cond)
        
        m_im = None
        if est_mask:
            # use probability to estimate mask region, combine with last layer and dilate
            mask_mapping_t = self.prob_to_mask(mapping_pts_left, prob_mapping, roi, (mapping.size(2), mapping.size(3)))
            mask_arr = np.repeat(np.expand_dims((mask_mapping_t.cpu().numpy().transpose(1, 2, 0)[:, :, 0]*255).astype(np.uint8), -1), 3, axis=-1)
            m_im = Image.fromarray(mask_arr)
            m_im = util.dilate(m_im, patchsize)
        
        if vfc_ratio_mapping < self.vfc_ratio_bound:
            print("region:", roi, "vfc_ratio_mapping:", vfc_ratio_mapping, "not updating")
            return mapping, m_im
        else:
            return mapping_interp, m_im
    
    def vfc_progressive_interp_corner(self, mapping, mask_lt, roi, search_box_radius, left_to_right=True):
        ### progressive interpolation of NNF using VFC
        step = search_box_radius // 2 # search_box_radius as the step_length of progressive interpolation
        if left_to_right:
            src_idx = 0
        else:
            src_idx = 1 # right_to_left mapping
        roi_width = roi[1] - roi[0]
        step_total = roi_width // step
        upper_left, bottom_right = (roi[0], roi[2]), (roi[1], roi[3])

        # upper_left to bottom right
        mapping_ul = mapping.clone()
        for i in range(1, step_total):
            # region: checkboard distance to upper_left corner < i*step 
            chkd_threshold = min(roi_width, step*i)
            region = (upper_left[0], upper_left[0]+chkd_threshold, upper_left[1], upper_left[1]+chkd_threshold)
            if region[1] > mapping.size(2) or region[3] > mapping.size(3):
                continue # region out of bound(because of forcing width==height for roi)
            mapping_ul, mask_p = self.vfc_local_interp(mapping_ul, region, mask_cond=mask_lt, est_mask=False)
            
        # bottom right to upper left
        mapping_br = mapping.clone()
        for i in range(1, step_total):
            chkd_threshold = min(roi_width, step*i)
            region = (bottom_right[0]-i*step, bottom_right[0], bottom_right[1]-i*step, bottom_right[1])
            if region[0] < 0 or region[2] < 0:
                continue
            mapping_br, mask_p = self.vfc_local_interp(mapping_br, region, mask_cond=mask_lt, est_mask=False)
            
        # aggregate both directions
        # mapping_diff = torch.mean(torch.abs(mapping_ul - mapping_br)).item()
        # print("diff of mapping:", mapping_diff)
        # mapping_interp = self.avg_mapping([mapping_ul, mapping_br], roi, mask_lt)
        mapping_interp = util.weight_mapping([mapping_ul, mapping_br], [upper_left, bottom_right], roi, mask_lt)
        return mapping_interp


    def vfc_progressive_interp(self, mapping, refined_correspondence, mask_lt, roi, search_box_radius, left_to_right=True):
        ### progressive interpolation of NNF using VFC
        step = 2 # search_box_radius // 2 # search_box_radius as the step_length of progressive interpolation
        if left_to_right:
            src_idx = 0
        else:
            src_idx = 1 # right_to_left mapping
        roi_width = roi[1] - roi[0]
        step_total = roi_width // step
        ul, br = (roi[0], roi[2]), (roi[1], roi[3])
        ur, bl = (roi[0], roi[3]), (roi[1], roi[2])
        max_region = (0, mapping.size(2), 0, mapping.size(3))
        
        if self.debug:
            print("ROI_p:", roi, "roi_width:", roi_width, "step:", step, "upper_left:", ul, "bottom_rigth:", br)
        
        # pick anchors
        anchors = []
        bd_d_i = util.chkd_roi(refined_correspondence[0], roi)
        roi_buddies_idx = [i for i in range(len(bd_d_i)) if bd_d_i[i] <= search_box_radius//2]
        if len(roi_buddies_idx) == 0:
            anchors = random.sample(refined_correspondence[0], min(2, len(refined_correspondence[0])))
        else:
            random.shuffle(roi_buddies_idx)
            init_anchors = [ul, br, ur, bl]
            init_flags = [True, True, True, True]
            for j in roi_buddies_idx:
                bd = refined_correspondence[0][j]
                chkd_bd_vec = util.chkd_vec(init_anchors, bd)
                for i, d in enumerate(chkd_bd_vec):
                    if d <= search_box_radius//2:
                        init_anchors.pop(i)
                        init_flags[i] = False 
                        anchors.append(bd)
                # all anchors picked
                if sum(init_flags) == 0:
                    break
        # no anchor picked
        if len(anchors) == 0:
            anchors_id = random.sample(roi_buddies_idx, min(2, len(roi_buddies_idx)))
            anchors = [refined_correspondence[0][i] for i in anchors_id]
            if len(anchors) == 0:
                anchors = [ul, br, ur, bl]

        # progressive interpolation centering each anchor
        mapping_interp_list = []
        for bd in anchors:
            mapping_i = mapping.clone()
            mapping_progress, region_list = [], []
            for i in range(1, step_total):
                interp_radius = min(roi_width, step*i)
                interp_region = (bd[0]-interp_radius, bd[0]+interp_radius, bd[1]-interp_radius, bd[1]+interp_radius)
                interp_region = util.clip_region(interp_region, max_region)
                region_list.append(interp_region)
                mapping_i, _ = self.vfc_local_interp(mapping_i, interp_region, mask_cond=mask_lt, est_mask=False)
                mapping_progress.append(mapping_i)
                
            if self.debug:
                h, w = mapping.size(2), mapping.size(3)
                map_arr_list = []
                for j, k in zip(mapping_progress, region_list):
                    map_arr = util.flow_to_color(util.mapping_to_uv(j)) * np.expand_dims(np.array(util.gen_roi_mask(h, w, k))>0, -1)
                    map_arr = util.draw_points_dot(map_arr, [bd])
                    map_arr_list.append(map_arr)

                map_progressive_arr = np.concatenate(map_arr_list, axis=1)

                plt.figure(figsize=(20, 5))
                plt.axis("off")
                plt.imshow(Image.fromarray(map_progressive_arr.astype(np.uint8)))
                plt.show()
            mapping_interp_list.append(mapping_i)

        # aggregate interpolated mappings
        mapping_interp = util.anchor_weight_mapping(mapping_interp_list, anchors, roi, mask_lt)
        if self.debug:
            print("progressive interpolation anchors:", anchors)
            print("mapping_in, mapping_interp, mapping_anchors")
            map_concat = []
            for j in [mapping, mapping_interp]:
                map_concat.append(util.flow_to_color(util.mapping_to_uv(j)))
            for j in mapping_interp_list:
                map_concat.append(util.flow_to_color(util.mapping_to_uv(j)))

            mask_arr = util.vis_mask(mask_lt)
            mask_arr_corr_vfc = util.draw_points_dot(mask_arr.copy(), anchors)
            map_concat.append(mask_arr_corr_vfc)
            
            final_map = np.concatenate(map_concat, axis=1)
            
            plt.figure(figsize=(20, 10))
            plt.axis("off")
            plt.imshow(Image.fromarray(final_map))
            plt.show()

        return mapping_interp
    
    
    def correspondence_to_mapping(self, correspondence, F_A, F_Am, F_Am_gt, F_Bm, F_B, patch_size, initial_map_a_to_b, initial_map_a_to_b_gt, initial_map_b_to_a,
                                  search_box_radius, roi=None, deepest_level=False, L=0, mask_lt=None):
        assert(F_A.size() == F_Bm.size())
        assert(F_Am.size() == F_B.size())

        F_Bm_normalized = FM.normalize_per_pix(F_Bm)
        F_Am_normalized = FM.normalize_per_pix(F_Am)
        F_Am_gt_normalized = FM.normalize_per_pix(F_Am_gt)
        
        # init mapping NNF, mapping: Tensor([1, 2, A.size(2), A.size(3)), F_Am(complete), F_Am(corrupted)
        a_to_b = self.find_mapping(F_Am_normalized, F_Bm_normalized, patch_size, initial_map_a_to_b, search_box_radius)
        a_to_b_gt = self.find_mapping(F_Am_gt_normalized, F_Bm_normalized, patch_size, initial_map_a_to_b_gt, search_box_radius)
        b_to_a = self.find_mapping(F_Bm_normalized, F_Am_normalized, patch_size, initial_map_b_to_a, search_box_radius)
        # a_to_b_old, b_to_a_old = a_to_b.clone(), b_to_a.clone()
        
        # estimate outlier region as predicted mask, shrink ROI region to roi_p(max bound of estimated mask)
        if deepest_level:
            a_to_b_roi_interp = a_to_b.clone()
            mask_p = self.Tensor(1, F_A.size(2), F_A.size(3)).fill_(0.0)
            roi_p = roi
        else:
            a_to_b_roi_interp_list, mask_im_list = [], []
            for _ in range(4):
                a_to_b_roi_interp, mask_im = self.vfc_local_interp(a_to_b, roi, est_mask=True, patchsize=patch_size[0])
                a_to_b_roi_interp_list.append(a_to_b_roi_interp)
                mask_im_list.append(mask_im.convert('L'))
            a_to_b_roi_interp = util.avg_mapping(a_to_b_roi_interp_list, roi, mask_lt)
            mask_im = util.combine_mask(mask_im_list)
            mask_p = util.make_mask(mask_im).cuda()[0, :, :].unsqueeze(0)
            roi_p = util.get_box(mask_im.convert('L'))
            roi_p = (max(roi[0], roi_p[0]-search_box_radius), min(roi[1], roi_p[1]+search_box_radius), max(roi[2], roi_p[2]-search_box_radius), min(roi[3], roi_p[3]+search_box_radius))
            if roi_p[1] < roi_p[0] or roi_p[3] < roi_p[2]:
                mask_p = mask_lt
                roi_p = roi
        self.a_to_b_roi_interp = a_to_b_roi_interp
        # F_Am, F_Bm: for finding buddies; F_Am, F_Bm: for finding b_to_a mapping
        # F_A, F_B: for activation filtering
        # find initial best buddies, only in or near ROI region(not roi_p)
        refined_correspondence = self.find_neural_best_buddies_roi(correspondence, F_A, F_B, a_to_b, b_to_a, search_box_radius, roi, L, deepest_level)

        while True:
            # skip deepest_level
            if deepest_level:
                a_to_b_interp = a_to_b.clone()
                break
            
            # use a_to_b(Am_in) instead of a_to_b(Am) for interpolation; use predicted mask and predicted ROI
            a_to_b_interp = self.vfc_progressive_interp(a_to_b, refined_correspondence, mask_p, roi_p, search_box_radius)
                
            break
        self.refined_correspondence, self.mask_p, self.roi_p = refined_correspondence, mask_p, roi_p
        return [refined_correspondence, a_to_b, a_to_b_interp, a_to_b_gt, b_to_a]
    
    def transfer_style_local(self, F_A, F_A_gt, F_B, patch_size, image_width, mapping_a_to_b_interp, mapping_a_to_b, mapping_a_to_b_gt, mapping_b_to_a, L, mask_lt=None):
        # input features
        self.model.set_input(self.A)
        FL_1A_in = self.model.forward(level = L-1).data # Style_A, Structure_A, x2
        self.model.set_input(self.B)
        FL_1B = self.model.forward(level = L-1).data # Style_B, Structure_B, x2
        
        # features with swapped structures
        F_A_warped = self.warp(F_B.size(), F_A, patch_size, mapping_b_to_a) # Style_A, Structure_B
        F_B_warped = self.warp(F_A.size(), F_B, patch_size, mapping_a_to_b) # Style_B, Structure_A_in
        F_B_warped_gt = self.warp(F_A.size(), F_B, patch_size, mapping_a_to_b_gt)
        RL_1B_in = self.model.deconve(F_B_warped, image_width, L, L-1, print_errors=False).data # Style_B, Structure_A_in, x2
        RL_1A = self.model.deconve(F_A_warped, image_width, L, L-1, print_errors=False).data # Style_A, Structure_B, x2
        
        # B features with Completed A structure
        RL_1B_gt = self.model.deconve(F_B_warped_gt, image_width, L, L-1, print_errors=False).data
        
        return [FL_1A_in, FL_1B, RL_1B_in, RL_1B_gt, RL_1A]
    
    def run(self, A, B, A_in, mask):
        assert(A.size() == B.size())
        image_width = A.size(3)

        self.A_gt = self.Tensor(A.size()).copy_(A)
        self.A = self.Tensor(A.size()).copy_(A_in)
        self.B = self.Tensor(B.size()).copy_(B)


        self.model.set_input(self.A)
        F_A = self.model.forward(level = self.L_start).data
        self.model.set_input(self.B)
        F_B = self.model.forward(level = self.L_start).data
        self.model.set_input(self.A_gt)
        F_A_gt = self.model.forward(level = self.L_start).data
        F_Am, F_Am_gt = F_A.clone(), F_A.clone()
        F_Bm = F_B.clone()
        if self.debug:
            RL_1B_in, RL_1B_gt = F_A.clone(), F_A.clone()
            RL_1A = F_B.clone()

        initial_map_a_to_b = self.identity_map(F_B.size())
        initial_map_b_to_a = initial_map_a_to_b.clone()
        initial_map_a_to_b_gt = initial_map_a_to_b.clone()

        for L in range(self.L_start, self.L_final-1, -1):
            patch_size = self.patch_size_list[L-1]
            search_box_radius = self.search_box_radius_list[L-1]
            size = F_B.shape[-2:]
            mask_l = util.resize_dilate(mask.copy(), size, 0)
            mask_lt = util.make_mask(mask_l).cuda()
            roi_l = util.get_box(mask_l)
            # dilate mask area by search_box_radius as roi region
            roi_l = (max(0, roi_l[0]-search_box_radius//2), min(size[0], roi_l[1]+search_box_radius//2), max(0, roi_l[2]-search_box_radius//2), min(size[1], roi_l[3]+search_box_radius//2))
            # assert (roi_l[1]-roi_l[0]) == (roi_l[3]-roi_l[2])
            
            
            if L == self.L_start:
                deepest_level = True
                correspondence = []

            else:
                # mix A and B style, align B to A and A to B
                # interpolate next layer mask
                mask_lt = functional.interpolate(mask_lt.unsqueeze(0), F_A.size()[-2:])[0]
                if L != 0:
                    initial_map_a_to_b = self.upsample_mapping(mapping_a_to_b)
                    initial_map_a_to_b_gt = self.upsample_mapping(mapping_a_to_b_gt)
                    initial_map_b_to_a = self.upsample_mapping(mapping_b_to_a)
                F_Am = (RL_1B_in + F_A) * 0.5 # Style_A, Structure_A (masked region are mixed)
                F_Bm = (F_B + RL_1A) * 0.5 # Style_A+B, Structure_B
                F_Am_gt = (F_A_gt + RL_1B_gt) * 0.5
                deepest_level = False
            
            if self.debug:
                print("next layer features")       
                im_concat = []
                print("F_A_gt:", F_A_gt.shape, "Up Mask:", mask_lt.shape)
                print("F_A_gt, F_A, F_Am, F_B, F_Bm, (F_A-F_A_gt), (F_A-F_A_gt)")
                for j in [F_A_gt, F_A, F_Am, F_B, F_Bm]:
                    im_concat.append(util.feature2images(j))
                    
                # diff between completion from last layer and current upsampled mapping warp
                im_concat.append(util.map2image(FM.error_map(F_A, F_A_gt)))
                im_concat.append(util.map2image(FM.error_map(F_Am, F_Am_gt)))
                    
                final_im = np.concatenate(im_concat, axis=1)
                plt.figure(figsize=(20, 5))
                plt.axis("off")
                plt.imshow(Image.fromarray(final_im))
                plt.show()

            print("Finding mapping for the " + str(L) + "-th level")
            [correspondence, mapping_a_to_b, mapping_a_to_b_interp, mapping_a_to_b_gt, mapping_b_to_a] = self.correspondence_to_mapping(correspondence, F_A, F_Am, F_Am_gt, F_Bm, F_B,  patch_size, 
                                                                                              initial_map_a_to_b, initial_map_a_to_b_gt, initial_map_b_to_a, search_box_radius, 
                                                                                              roi=roi_l, 
                                                                                              deepest_level=deepest_level, L=L, mask_lt=mask_lt)

            
            if self.debug:
                print("after intepolation")
                print("ROI_l:", roi_l, "mapping shape", mapping_a_to_b.shape, "Feat_shape:", F_A.shape) #  "Est. Mask:", mask_p.shape,
                
                map_concat = []
                for j in [mapping_a_to_b_gt, mapping_a_to_b, mapping_a_to_b_interp, mapping_b_to_a]:
                    map_concat.append(util.flow_to_color(util.mapping_to_uv(j)))
                mask_arr = util.vis_mask(mask_lt)
                map_concat.append(mask_arr)
                F_A_corr = util.draw_points_dot(util.feature2images(F_A), correspondence[0])
                F_B_corr = util.draw_points_dot(util.feature2images(F_B), correspondence[1])
                map_concat.append(F_A_corr)
                map_concat.append(F_B_corr)

                final_map = np.concatenate(map_concat, axis=1)
                print("mapping_a_to_b_gt, mapping_a_to_b, mapping_a_to_b_interp, mapping_b_to_a")
                plt.figure(figsize=(20, 5))
                plt.axis("off")
                plt.imshow(Image.fromarray(final_map))
                plt.show()
  
            if L == (self.L_final):
                print("final")
                F_Am = (RL_1B_in + F_A) * 0.5
                F_Bm = (F_B + RL_1A)*0.5
                
                self.final_list = [F_A, F_A_gt, F_B, RL_1A, F_Bm, F_Am, RL_1B_in, mapping_a_to_b_interp, mapping_a_to_b, mapping_a_to_b_gt, mapping_b_to_a, correspondence]
                
                return self.final_list
            # next layer featuers
            [F_A, F_B, RL_1B_in, RL_1B_gt, RL_1A] = self.transfer_style_local(F_A, F_A_gt, F_B, patch_size, image_width, mapping_a_to_b_interp, mapping_a_to_b, mapping_a_to_b_gt, mapping_b_to_a, L, mask_lt=mask_lt)
            self.model.set_input(self.A_gt)
            F_A_gt = self.model.forward(level = L-1).data
            