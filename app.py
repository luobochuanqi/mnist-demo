import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import transforms
from layers import SimpleNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "mnist_model.pth"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

model = SimpleNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()


def preprocess_image(image_dict):
    if image_dict is None or "composite" not in image_dict:
        return None

    image = image_dict["composite"]
    image = image.convert("L")
    image = image.resize((28, 28))
    image = np.array(image)

    image = 255 - image
    image = image.astype(np.float32) / 255.0

    mean = 0.1307
    std = 0.3081
    image = (image - mean) / std

    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return image


def predict_digit(image_dict):
    if image_dict is None:
        return "请在画布上写一个数字", None

    preprocessed = preprocess_image(image_dict)
    if preprocessed is None:
        return "无法处理图像", None

    with torch.no_grad():
        output = model(preprocessed.to(DEVICE))
        probabilities = F.softmax(output, dim=1)
        predicted = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted].item()

    probs_array = probabilities[0].cpu().numpy()
    probs_df = pd.DataFrame(
        {"Digit": list(range(10)), "Probability": [float(p) for p in probs_array]}
    )

    result = f"预测结果：{predicted} (置信度：{confidence:.2%})"
    return result, probs_df


def clear_canvas():
    return None, "画布已清空", None


with gr.Blocks() as app:
    gr.Markdown("# 🖊️ MNIST 手写数字识别")
    gr.Markdown("在下方的画布上写一个数字 (0-9)，模型会实时预测您写的数字。")

    with gr.Row():
        with gr.Column():
            canvas = gr.Sketchpad(
                type="pil",
                label="画布",
                height=300,
                width=300,
            )

            with gr.Row():
                predict_btn = gr.Button("🔮 预测", variant="primary")
                clear_btn = gr.Button("🧹 清空画布", variant="secondary")

        with gr.Column():
            output_text = gr.Textbox(label="预测结果", interactive=False)
            output_plot = gr.BarPlot(label="各类别概率", x="Digit", y="Probability")

    predict_btn.click(
        fn=predict_digit, inputs=[canvas], outputs=[output_text, output_plot]
    )

    clear_btn.click(
        fn=clear_canvas, inputs=[], outputs=[canvas, output_text, output_plot]
    )


if __name__ == "__main__":
    print(f"加载模型：{MODEL_PATH}")
    print(f"运行设备：{DEVICE}")
    print("启动 Gradio 应用...")
    app.launch(theme=gr.themes.Soft())
