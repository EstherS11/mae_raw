import torch

org_templates = [
    'a bad photo of a {}.',
    'a bad photo of the {}.',
    'a black and white photo of a {}.',
    'a black and white photo of the {}.',
    'a blurry photo of a {}.',
    'a blurry photo of the {}.',
    'a bright photo of a {}.',
    'a bright photo of the {}.',
    'a close-up photo of a {}.',
    'a close-up photo of the {}.',
    'a cropped photo of a {}.',
    'a cropped photo of the {}.',
    'a dark photo of a {}.',
    'a dark photo of the {}.',
    'a good photo of a {}.',
    'a good photo of the {}.',
    'a jpeg corrupted photo of a {}.',
    'a jpeg corrupted photo of the {}.',
    'a low resolution photo of a {}.',
    'a low resolution photo of the {}.',
    'a photo of a cool {}.',
    'a photo of a large {}.',
    'a photo of a small {}.',
    'a photo of a {}.',
    'a photo of my {}.',
    'a photo of one {}.',
    'a photo of the cool {}.',
    'a photo of the large {}.',
    'a photo of the small {}.',
    'a photo of the {}.',
    'there is a {} in the scene.',
    'there is the {} in the scene.',
    'this is a {} in the scene.',
    'this is one {} in the scene.',
    'this is the {} in the scene.']
base_templates = org_templates + [
    "a photo of the {} for visual inspection",
    "a photo of a {} for visual inspection",
    "a photo of the {} for anomaly detection",
    "a photo of a {} for anomaly detection",]
industrial_templates = [
    "a cropped industrial photo of the {}",
    "a cropped industrial photo of a {}",
    "a close-up industrial photo of a {}",
    "a close-up industrial photo of the {}",
    "a bright industrial photo of a {}",
    "a bright industrial photo of the {}",
    "a dark industrial photo of the {}",
    "a dark industrial photo of a {}",
    "a jpeg corrupted industrial photo of a {}",
    "a jpeg corrupted industrial photo of the {}",
    "a blurry industrial photo of the {}",
    "a blurry industrial photo of a {}",
    "an industrial photo of a {}",
    "an industrial photo of the {}",
    "an industrial photo of a small {}",
    "an industrial photo of the small {}",
    "an industrial photo of a large {}",
    "an industrial photo of the large {}",
    "an industrial photo of the {} for visual inspection",
    "an industrial photo of a {} for visual inspection",
    "an industrial photo of the {} for anomaly detection",
    "an industrial photo of a {} for anomaly detection", ]
image_templates = [
    "a cropped industrial image of the {}",
    "a cropped industrial image of a {}",
    "a close-up industrial image of a {}",
    "a close-up industrial image of the {}",
    "a bright industrial image of a {}",
    "a bright industrial image of the {}",
    "a dark industrial image of the {}",
    "a dark industrial image of a {}",
    "a jpeg corrupted industrial image of a {}",
    "a jpeg corrupted industrial image of the {}",
    "a blurry industrial image of the {}",
    "a blurry industrial image of a {}",
    "an industrial image of a {}",
    "an industrial image of the {}",
    "an industrial image of a small {}",
    "an industrial image of the small {}",
    "an industrial image of a large {}",
    "an industrial image of the large {}",
    "an industrial image of the {} for visual inspection",
    "an industrial image of a {} for visual inspection",
    "an industrial image of the {} for anomaly detection",
    "an industrial image of a {} for anomaly detection", ]
manufacturing_templates = [
    "a cropped manufacturing image of the {}",
    "a cropped manufacturing image of a {}",
    "a close-up manufacturing image of a {}",
    "a close-up manufacturing image of the {}",
    "a bright manufacturing image of a {}",
    "a bright manufacturing image of the {}",
    "a dark manufacturing image of the {}",
    "a dark manufacturing image of a {}",
    "a jpeg corrupted manufacturing image of a {}",
    "a jpeg corrupted manufacturing image of the {}",
    "a blurry manufacturing image of the {}",
    "a blurry manufacturing image of a {}",
    "a manufacturing image of a {}",
    "a manufacturing image of the {}",
    "a manufacturing image of a small {}",
    "a manufacturing image of the small {}",
    "a manufacturing image of a large {}",
    "a manufacturing image of the large {}",
    "a manufacturing image of the {} for visual inspection",
    "a manufacturing image of a {} for visual inspection",
    "a manufacturing image of the {} for anomaly detection",
    "a manufacturing image of a {} for anomaly detection", ]
textural_templates = [
    "a cropped textural photo of the {}",
    "a cropped textural photo of a {}",
    "a close-up textural photo of a {}",
    "a close-up textural photo of the {}",
    "a bright textural photo of a {}",
    "a bright textural photo of the {}",
    "a dark textural photo of the {}",
    "a dark textural photo of a {}",
    "a jpeg corrupted textural photo of a {}",
    "a jpeg corrupted textural photo of the {}",
    "a blurry textural photo of the {}",
    "a blurry textural photo of a {}",
    "a textural photo of a {}",
    "a textural photo of the {}",
    "a textural photo of a small {}",
    "a textural photo of the small {}",
    "a textural photo of a large {}",
    "a textural photo of the large {}",
    "a textural photo of the {} for visual inspection",
    "a textural photo of a {} for visual inspection",
    "a textural photo of the {} for anomaly detection",
    "a textural photo of a {} for anomaly detection", ]
surface_templates = [
    "a cropped surface photo of the {}",
    "a cropped surface photo of a {}",
    "a close-up surface photo of a {}",
    "a close-up surface photo of the {}",
    "a bright surface photo of a {}",
    "a bright surface photo of the {}",
    "a dark surface photo of the {}",
    "a dark surface photo of a {}",
    "a jpeg corrupted surface photo of a {}",
    "a jpeg corrupted surface photo of the {}",
    "a blurry surface photo of the {}",
    "a blurry surface photo of a {}",
    "a surface photo of a {}",
    "a surface photo of the {}",
    "a surface photo of a small {}",
    "a surface photo of the small {}",
    "a surface photo of a large {}",
    "a surface photo of the large {}",
    "a surface photo of the {} for visual inspection",
    "a surface photo of a {} for visual inspection",
    "a surface photo of the {} for anomaly detection",
    "a surface photo of a {} for anomaly detection", ]


from copy import deepcopy
import torch.nn as nn

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])
class AnomalyCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model,tokenizer):
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)  ## 1
        self.n_ctx = 12  ## 12
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = 4 ## 4
        ctx_init_pos = ""
        ctx_init_neg = ""
        dtype = clip_model.transformer.get_cast_dtype()  ## fp32

        ctx_dim = clip_model.ln_final.weight.shape[0]   ## 768
        self.tokenizer = tokenizer
        
        self.classnames = classnames

        # self.state_normal_list = [
        #     "{}",
        # ]

        # self.state_anomaly_list = [
        #     "forged {}",
        # ]
        self.state_normal_list = [
            "Untouched {}",  # 未触动/原始的
            "Authentic {}",  # 真实的
            "Real {}",       # 真实的
            "Original {}",   # 原始的
            "Genuine {}",    # 真正的
        ]

        # 异常状态的描述模板
        self.state_anomaly_list = [
            "Manipulated {}",  # 操作过的
            "Forged {}",       # 伪造的
            "Fake {}",         # 假的
            "Altered {}",      # 修改过的
            "Tampered {}",     # 篡改的
        ]
        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            #初始化text成bpd编码
            prompt_pos = self.tokenizer(ctx_init_pos)
            prompt_neg = self.tokenizer(ctx_init_neg)
            with torch.no_grad():
                #生成相应的text embedding
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            #这些是去除出来EOS 和 # CLS, EOS， 获得可学习的textual prompt
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if True:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if True:
                print("Initializing class-specific contexts")
                #这里是cls是类的个数，n_ctx_pos代表learnable token的长度，ctx_dim表示prompt的dimension
                ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)  # torch.Size([1, 1, 12, 768])
                ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)  ### 12个可学习的token
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
        # self.compound_prompts_depth = 9  ## 9个深度复合文本？？
        # self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))  # 4 * 768
        #                                               for _ in range(self.compound_prompts_depth - 1)])  # 9
        # for single_para in self.compound_prompts_text:
        #     print("single_para", single_para.shape)
        #     nn.init.normal_(single_para, std=0.02)

        # single_layer = nn.Linear(ctx_dim, 896)
        # self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)


        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]  ## 【‘object’】
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]


        prompts_pos = [prompt_prefix_pos +  " " + template.format(name)+ "." for template in self.state_normal_list for name in classnames]
        prompts_neg = [prompt_prefix_neg +  " " + template.format(name)+ "." for template in self.state_anomaly_list for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
     
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(self.tokenizer(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(self.tokenizer(p_neg))  ## 1 * 77
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        #生成相应的text embedding
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape  ## torch.Size([1, 77, 768])
            print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)


        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :] )
        self.register_buffer("token_suffix_pos", embedding_pos[:, :,1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:,:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])

        n, d = tokenized_prompts_pos.shape  # 1,77
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)  ## 1,1,77

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        # tokenized_prompts = torch.cat([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)



    def forward(self, cls_id =None):
        
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        # print("shape", self.ctx_pos[0:1].shape, ctx_pos.shape)
        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        # print(prefix_pos.shape, prefix_neg.shape)

        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=2,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=2,
        )
        _, _, l, d = prompts_pos.shape  # 77,768
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)


        _, l, d = self.tokenized_prompts_pos.shape  ### 1 ,77
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1,  d)
        _, l, d = self.tokenized_prompts_neg.shape   # 1 ,77
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1,  d)
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim = 0)

        return prompts, tokenized_prompts, None
        # return prompts, tokenized_prompts, self.compound_prompts_text