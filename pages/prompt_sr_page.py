import gradio as gr
import random


def gen_sr(lora_name: str, decimals: str, integers: str):
    """
    Given lora_name and decimals, return a string containing all variations with decimals specified by decimals

    :param lora_name: "<lora:maimuro-nd-locon-epoch16:0.1>"
    :param decimals: "0.1, 0.3, 0.5, 0.8"
    :param integers: "10, 25, 50"   # not using it, keep it here for compatibility
    :return: "<lora:maimuro-nd-locon-epoch16:0.1>, <lora:maimuro-nd-locon-epoch16:0.3>, <lora:maimuro-nd-locon-epoch16:0.5>, <lora:maimuro-nd-locon-epoch16:0.8>"
    """
    lora_name = lora_name.split(":")[1]

    if decimals:
        decimal_list = [i.strip() for i in decimals.split(",")]

    else:
        integers_list = [i.strip() for i in integers.split(",")]
        decimal_list = ["{:.2f}".format(int(i) / 100) for i in integers_list]

    return_list = [f"<lora:{lora_name}:{decimal}>" for decimal in decimal_list]
    new_prompt = ", ".join(return_list)
    info = f"{len(return_list)} prompts generated"

    return new_prompt, info


def get_prompt_sr_page():

    lora_name = gr.Text(placeholder="<lora:maimuro-nd-locon-epoch16:1>", label="lora name")
    decimals = gr.Text(placeholder="0.1, 0.3, 0.5, 1", label="percentile(decimal)")
    integers = gr.Text(placeholder="0, 15, 50, 100", label="percentile(integers)")

    page = gr.Interface(
        gen_sr,
        inputs=[
            lora_name,
            decimals,
            integers
        ],
        outputs=[
            gr.TextArea(placeholder="new prompt"),
            gr.TextArea(placeholder="info")
        ],

    )

    return page


if __name__ == '__main__':
    # r = gen_sr("<lora:maimuro-nd-locon-epoch16:0.1>", "0.1, 0.3, 0.5, 0.8", "")
    r = gen_sr("<lora:maimuro-nd-locon-epoch16:0.1>", "", "10, 50, 100")

    print(r)
