import random

templates_lib = [
    "a photo of a [CLS].",
    "a picture of a [CLS].",
    "a sketch of a [CLS].",
    "a clipart of a [CLS].",
    "a painting of a [CLS].",
    "a sculpture of a [CLS].",
    "a doodle of a [CLS].",
    "art of the [CLS].",
    "a cartoon [CLS].",
    "a origami [CLS].",
    "a plushie [CLS].",
    "a plastic [CLS].",
    "a embroidered [CLS].",
    "a [CLS] in real world.",
    "a [CLS] in a video game.",
    "a bad photo of the [CLS].",
    "a bright photo of a [CLS].",
    "a corrupted photo of a [CLS].",
    "a dark photo of a [CLS].",
    "a good photo of a [CLS].",
    "a blurry photo of a [CLS].",
    "a low contrast photo of a [CLS].",
    "a high contrast photo of a [CLS].",
    "a photo of the [CLS] from the front.",
    "a photo of the [CLS] from the back.",
    "a photo of the [CLS] from the side.",
    "a photo of a [CLS] in the distance."
]


def get_text_prompts(cls_name, sample_times) -> list:
    if sample_times > len(templates_lib):
        raise ValueError(f"Requested {sample_times} samples, but only {len(templates_lib)} templates available")

    prompts_list = []
    selected_templates = random.sample(templates_lib, sample_times)
    for template in selected_templates:
        prompt = template.replace('[CLS]', cls_name)
        prompts_list.append(prompt)

    return prompts_list
