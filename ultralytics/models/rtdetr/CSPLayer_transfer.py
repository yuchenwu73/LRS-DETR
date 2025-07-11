import torch
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import CSPLayer
from ultralytics.nn.extra_modules.block import DEFBlock
from ultralytics.nn.extra_modules.prune_module import *

def transfer_weights_CSPLayer_v2_to_CSPLayer(CSPLayer_v2, CSPLayer):
    CSPLayer.cv2 = CSPLayer_v2.cv2
    CSPLayer.m = CSPLayer_v2.m

    state_dict = CSPLayer.state_dict()
    state_dict_v2 = CSPLayer_v2.state_dict()

    # Transfer cv1 weights from CSPLayer to cv0 and cv1 in CSPLayer_v2
    old_weight = state_dict['cv1.conv.weight']
    new_cv1 = Conv(c1=state_dict_v2['cv0.conv.weight'].size()[1],
                   c2=(state_dict_v2['cv0.conv.weight'].size()[0] + state_dict_v2['cv1.conv.weight'].size()[0]),
                   k=CSPLayer_v2.cv1.conv.kernel_size,
                   s=CSPLayer_v2.cv1.conv.stride)
    CSPLayer.cv1 = new_cv1
    CSPLayer.c1, CSPLayer.c2 = state_dict_v2['cv0.conv.weight'].size()[0], state_dict_v2['cv1.conv.weight'].size()[0]
    state_dict['cv1.conv.weight'] = torch.cat([state_dict_v2['cv0.conv.weight'], state_dict_v2['cv1.conv.weight']], dim=0)

    # Transfer cv1 batchnorm weights and buffers from CSPLayer to cv0 and cv1 in CSPLayer_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        state_dict[f'cv1.bn.{bn_key}'] = torch.cat([state_dict_v2[f'cv0.bn.{bn_key}'], state_dict_v2[f'cv1.bn.{bn_key}']], dim=0)

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict[key] = state_dict_v2[key]

    # Transfer all non-method attributes
    for attr_name in dir(CSPLayer_v2):
        attr_value = getattr(CSPLayer_v2, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(CSPLayer, attr_name, attr_value)

    CSPLayer.load_state_dict(state_dict)

def replace_CSPLayer_v2_with_CSPLayer(module):
    # for CSPLayer
    for name, child_module in module.named_children():
        if isinstance(child_module, CSPLayer_v2):
            # Replace CSPLayer with CSPLayer_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            csplayer = CSPLayer_infer(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights_CSPLayer_v2_to_CSPLayer(child_module, csplayer)
            setattr(module, name, csplayer)
        else:
            replace_CSPLayer_v2_with_CSPLayer(child_module)

def infer_shortcut(bottleneck):
    try:
        c1 = bottleneck.cv1.conv.in_channels
        c2 = bottleneck.cv2.conv.out_channels
        return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add
    except:
        return False

def transfer_weights_CSPLayer_to_CSPLayer_v2(CSPLayer, CSPLayer_v2):
    CSPLayer_v2.cv2 = CSPLayer.cv2
    CSPLayer_v2.m = CSPLayer.m

    state_dict = CSPLayer.state_dict()
    state_dict_v2 = CSPLayer_v2.state_dict()

    # Transfer cv1 weights from CSPLayer to cv0 and cv1 in CSPLayer_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from CSPLayer to cv0 and cv1 in CSPLayer_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(CSPLayer):
        attr_value = getattr(CSPLayer, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(CSPLayer_v2, attr_name, attr_value)

    CSPLayer_v2.load_state_dict(state_dict_v2)

def replace_CSPLayer_with_CSPLayer_v2(module):
    # Import the v2 classes at the top of the function to avoid naming conflicts
    # Make sure these imports match your actual module structure
    from ultralytics.nn.extra_modules.prune_module import CSPLayer_v2, DEFBlock_v2
    
    for name, child_module in module.named_children():
        if isinstance(child_module, DEFBlock):
            # Replace CSPLayer with CSPLayer_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            defblock_v2 = DEFBlock_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=1,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights_CSPLayer_to_CSPLayer_v2(child_module, defblock_v2)
            setattr(module, name, defblock_v2)
        elif isinstance(child_module, CSPLayer):
             # Replace CSPLayer with CSPLayer_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            csplayer_v2 = CSPLayer_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights_CSPLayer_to_CSPLayer_v2(child_module, csplayer_v2)
            setattr(module, name, csplayer_v2)
        else:
            replace_CSPLayer_with_CSPLayer_v2(child_module)