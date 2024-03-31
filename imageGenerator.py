import openai
import torch
from diffusers import StableDiffusionPipeline, LoraConfig
import argparse

# OpenAI APIキー
openai.api_key = "XXXXXXXXXX"

def generate_prompt(input_text, model="text-davinci-003", temperature=0.7):
    response = openai.Completion.create(
        model=model,
        prompt=input_text,
        temperature=temperature,
        max_tokens=100,
        n=1
    )
    return response.choices[0].text.strip()

def generate_images(prompt, num_images=20, model_path="CompVis/stable-diffusion-v-1-5"):
    pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")
    
    # ネガティブプロンプト用のLORA「EasyNegative」を使用
    lora_config = LoraConfig.from_config("lora_configs/stable-diffusion-v-1-5/easy_negative.json")
    pipeline.enable_lora(lora_config)

    images = []
    for _ in range(num_images):
        image = pipeline(prompt)["sample"][0]
        images.append(image)

    return images

# メイン処理
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion 1.5")
    parser.add_argument("--model_path", type=str, help="Path to the Stable Diffusion model", required=True)
    args = parser.parse_args()

    input_text = "日本語で入力された内容"
    prompt = generate_prompt(input_text)
    images = generate_images(prompt, model_path=args.model_path)

    # 画像保存
    for i, image in enumerate(images):
        image.save(f"generated_image_{i+1}.png")
