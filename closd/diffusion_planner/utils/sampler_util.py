import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from closd.diffusion_planner.utils.misc import wrapped_getattr
import joblib

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, guidance_type='text'):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        self.guidance_type = guidance_type


    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        if 'text' in self.guidance_type:
            y_uncond['text_uncond'] = True
        if 'target' in self.guidance_type:
            y_uncond['target_uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))

    def __getattr__(self, name, default=None):
        # this method is reached only if name is not in self.__dict__.
        return wrapped_getattr(self, name, default=None)

'''
goal: autoregressive 방식을 사용해 짧은 동작 생성 모델을 반복적으로 호출 
-> long sequence motion을 생성하는 autoregressive sampler. 

기본적으로 diffusion model은 fixed length motion(ex)60frame)을 생성하도록 훈련됨.
근데 이 샘플러를 쓰면 196프레임 이상의 긴 동작도 자연스럽게 붙여 만들 수 있음. 
'''
class AutoRegressiveSampler():
    def __init__(self, args, sample_fn, required_frames=196):
        
        self.sample_fn = sample_fn # motion을 한 조각 생성하는 함수(diffusion sampling 함수) 
        self.args = args
        self.required_frames = required_frames # 최종적으로 생성하고자 하는 전체 프레임 수 == motion length(ex) 196frames)
    
    def sample(self, model, shape, **kargs):
        bs = shape[0]

        '''
        (전체 프레임 길이 / 힌 번에 생성할 수 있는 길이) 
        +1 을 통해 올림?? 그거 함. 
        -> 몇 번 루프돌지 결정. 
        '''
        n_iterations = (self.required_frames // self.args.pred_len) + 1 

        samples_buf = []

        # 첫 번째 생성을 위한 씨앗 동작. 
        # 정지 상태 or 이전 동작의 마지막 포즈. 
        cur_prefix = deepcopy(kargs['model_kwargs']['y']['prefix'])  # init with data
        if self.args.autoregressive_include_prefix:
            samples_buf.append(cur_prefix)

        # 한 번의 루프에서 생성할 텐서의 크기.(pred_len 길이) 
        autoregressive_shape = list(deepcopy(shape))

        # 마지막 차원이 생성한 길이로 설정됨. 
        autoregressive_shape[-1] = self.args.pred_len


        for _ in range(n_iterations):
            cur_kargs = deepcopy(kargs) 

            #이전 단계에서 만든 동작의 끝부분을 현재 생성의 조건으로 넣어줌. -> 끊기지 않고 부드럽게 이어짐. 
            cur_kargs['model_kwargs']['y']['prefix'] = cur_prefix 

            '''
            전체 길이는 sample(모델이 뱉어낸 전체 결과물) = 입력된 prefix + 새로 생성된 예측
            pred_len: 이번 턴에 순수하게 새로 만들어내야 하는 길이. 
            context_len: 이전에 만들어진 동작 중에서, 다음 동작 생성을 위한 조건으로 사용되는 길이.(다음턴의 prefix가 됨)
            '''

            '''
            모델이 prefix보고 다음 동작 조각(segment)을 생성.
            모델에게 이 앞부분(prefix) 뒤에 자연스러운 동작을 붙여보라고 시킴. 
            -> 결과(sample): 기존 prefix+ pred_len 길이의 새 동작이 합쳐진 것([기존 prefix(과거) | 새 예측(pred; 미래)])
            '''
            sample = self.sample_fn(model, autoregressive_shape, **cur_kargs)

            '''
            결과인 sample 중에서 순수하게 새로 만든 부분(pred)만 잘라서 sample_buf에 저장. 
            [..., -self.args.pred_len:]
            ... : 마지막 차원을 제외한 앞에 있는 차원은 모두 포함해라. 
            - : 뒤에서부터 세었을 때 해당 위치. 
            : : ~부터 끝까지. 
            - self.args.pred_len : 뒤에서 pred_len 번째 요소부터 맨 끝까지 잘라 <=> pred_len 길이만큼 자름.
            '''
            samples_buf.append(sample.clone()[..., -self.args.pred_len:])

            # 방금 만든 동작의 가장 뒷부분만큼 잘라냄. 이게 cur_prefix가 됨. -> 다음 루프에서 조건으로 사용됨.
            cur_prefix = sample.clone()[..., -self.args.context_len:]  # update

        # 조각난 동작들을 시간 축으로 이어붙여 하나의 긴 동작으로 만듦. 
        # 루프 땜에 길이가 남을 수 있으니, 필요한 만큼 잘라서 반환.
        full_batch = torch.cat(samples_buf, dim=-1)[..., :self.required_frames]  # 200 -> 196
        return full_batch