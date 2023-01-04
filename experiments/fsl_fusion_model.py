# from autogluon.multimodal.registry import Registry
#
# automm_fewshot_presets = Registry("automm_fewshot_presets")
#
# @automm_fewshot_presets.register()
# def default():
#     return {
#         "model.names": [
#             "categorical_mlp",
#             "numerical_mlp",
#             "timm_image",
#             "hf_text",
#             "fusion_mlp",
#             "ner_text",
#             "fusion_ner",
#         ],
#         "model.hf_text.checkpoint_name": "google/electra-base-discriminator",
#         "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
#         "env.num_workers": 2,
#     }
#
#
# @automm_fewshot_presets.register()
# def clip_swin_large_fusion():
#     return {
#         "model.names": [
#             "clip",
#             "timm_image",
#             "fusion_mlp"
#         ],
#         "model.clip.checkpoint_name": "openai/clip-vit-large-patch14-336",
#         "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224",
#         "model.clip.max_text_len": 0,
#         "env.num_workers": 2,
#     }
#
#
# def get_automm_fewshot_model(presets: str):
#     """
#     Map a AutoMM preset string to its config including a basic config and an overriding dict.
#     Parameters
#     ----------
#     presets
#         Name of a preset.
#     Returns
#     -------
#     basic_config
#         The basic config of AutoMM.
#     overrides
#         The hyperparameter overrides of this preset.
#     """
#     presets = presets.lower()
#     if presets in automm_fewshot_presets.list_keys():
#         overrides = automm_fewshot_presets.create(presets)
#     else:
#         raise ValueError(
#             f"Provided preset '{presets}' is not supported. " f"Consider one of these: {automm_fewshot_presets.list_keys()}"
#         )
#
#     return overrides
