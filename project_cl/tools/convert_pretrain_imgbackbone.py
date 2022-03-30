import torch

# if __name__ == '__main__':
#     # state_dict = torch.load('checkpoints/densecl_r50_coco_1600ep.pth')
#     state_dict = torch.load('checkpoints/moco_r50_v2-e3b0c442.pth')
#     new_state_dict = {}
#     key_param = state_dict['state_dict']
#     for key, value in key_param.items():
#         new_key = 'img_backbone.' + key
#         new_state_dict[new_key] = value

#     for key, value in new_state_dict.items():
#         print(key)
        
#     # torch.save(new_state_dict, 'checkpoints/densecl_r50_coco_1600ep_convert.pth')
#     torch.save(new_state_dict, 'checkpoints/moco_r50_v2-e3b0c442_convert.pth')

# if __name__ == '__main__':
#     state_dict = torch.load('checkpoints/waymo_ep50_without_imgbackbone.pth')
#     new_state_dict = {}
#     key_param = state_dict['state_dict']
#     for key, value in key_param.items():
#         new_key = 'backbone.' + key
#         new_state_dict[new_key] = value

#     for key, value in new_state_dict.items():
#         print(key)
#     torch.save(new_state_dict, 'checkpoints/waymo_ep50_with_backbone.pth')

if __name__ == '__main__':
    state_dict = torch.load('nfs/lzy/waymoExps/ablation_waymo/pretrain_onefive/epoch_10.pth')
    new_state_dict = {}
    key_param = state_dict['state_dict']
    # key_param = state_dict
    for key, value in key_param.items():
        _keys = key.split(".")
        if _keys[0] == 'img_backbone':
            new_key = ""
            for i in _keys[1:]:
                new_key = new_key + i + "."
            new_key = new_key[:-1]
            new_key = 'backbone.' + new_key
            new_state_dict[new_key] = value

    for key, value in new_state_dict.items():
        print(key)
    torch.save(new_state_dict, 'checkpoints/mono3d_waymo_onefive.pth')



    