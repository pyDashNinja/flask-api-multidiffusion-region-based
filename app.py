import gradio as gr
import numpy as np
import cv2
from PIL import Image
import torch
import base64
import requests 
import random
import os
from io import BytesIO
from region_control import MultiDiffusion, get_views, preprocess_mask, seed_everything
from sketch_helper import get_high_freq_colors, color_quantization, create_binary_matrix
MAX_COLORS = 12
os.environ['SPACE_ID'] = "weizmannscience/multidiffusion-region-based"

sd = MultiDiffusion("cuda", "2.1")
is_shared_ui = True if "weizmannscience/multidiffusion-region-based" in os.environ['SPACE_ID'] else False
is_gpu_associated = True if torch.cuda.is_available() else False
canvas_html = "<div id='canvas-root' style='max-width:400px; margin: 0 auto'></div>"
load_js = """
async () => {
const url = "https://huggingface.co/datasets/radames/gradio-components/raw/main/sketch-canvas.js"
fetch(url)
  .then(res => res.text())
  .then(text => {
    const script = document.createElement('script');
    script.type = "module"
    script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
    document.head.appendChild(script);
  });
}
"""

get_js_colors = """
async (canvasData) => {
  const canvasEl = document.getElementById("canvas-root");
  return [canvasEl._data]
}
"""

set_canvas_size ="""
async (aspect) => {
  if(aspect ==='square'){
    _updateCanvas(512,512)
  }
  if(aspect ==='horizontal'){
    _updateCanvas(768,512)
  }
  if(aspect ==='vertical'){
    _updateCanvas(512,768)
  }
}
"""

def process_sketch(canvas_data, binary_matrixes):
  binary_matrixes.clear()
  base64_img = canvas_data['image']
  image_data = base64.b64decode(base64_img.split(',')[1])
  image = Image.open(BytesIO(image_data)).convert("RGB")
  im2arr = np.array(image)
  colors = [tuple(map(int, rgb[4:-1].split(','))) for rgb in canvas_data['colors']]
  colors_fixed = []
  for color in colors:
    r, g, b = color
    if any(c != 255 for c in (r, g, b)):
      binary_matrix = create_binary_matrix(im2arr, (r,g,b))
      binary_matrixes.append(binary_matrix)
      colors_fixed.append(gr.update(value=f'<div style="display:flex;align-items: center;justify-content: center"><img width="20%" style="margin-right: 1em" src="file/{binary_matrix}" /><div class="color-bg-item" style="background-color: rgb({r},{g},{b})"></div></div>'))
  visibilities = []
  colors = []
  for n in range(MAX_COLORS):
    visibilities.append(gr.update(visible=False))
    colors.append(gr.update(value=f'<div class="color-bg-item" style="background-color: black"></div>'))
  for n in range(len(colors_fixed)):
    visibilities[n] = gr.update(visible=True)
    colors[n] = colors_fixed[n]
  return [gr.update(visible=True), binary_matrixes, *visibilities, *colors]

def process_generation(model, binary_matrixes, boostrapping, aspect, steps, seed, master_prompt, negative_prompt, *prompts):
    global sd
    if(model != "stabilityai/stable-diffusion-2-1-base"):
      sd = MultiDiffusion("cuda", model)
    if(seed == -1):
      seed = random.randint(1, 2147483647)
    seed_everything(seed)   
    dimensions = {"square": (512, 512), "horizontal": (768, 512), "vertical": (512, 768)}
    width, height = dimensions.get(aspect, dimensions["square"])  
    
    clipped_prompts = prompts[:len(binary_matrixes)]
    prompts = [master_prompt] + list(clipped_prompts)
    neg_prompts = [negative_prompt] * len(prompts)
    fg_masks = torch.cat([preprocess_mask(mask_path, height // 8, width // 8, "cuda") for mask_path in binary_matrixes])
    bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
    bg_mask[bg_mask < 0] = 0
    masks = torch.cat([bg_mask, fg_masks])
    print(masks.size())
    print(prompts, "----", neg_prompts)
    image = sd.generate(masks, prompts, neg_prompts, height, width, steps, bootstrapping=boostrapping)
    return(image)

css = '''
#color-bg{display:flex;justify-content: center;align-items: center;}
.color-bg-item{width: 100%; height: 32px}
#main_button{width:100%}
<style>
'''

with gr.Blocks(css=css) as demo:
  binary_matrixes = gr.State([])
  gr.Markdown('''## Control your Stable Diffusion generation with Sketches (_beta_)
  A beta version demo of [MultiDiffusion](https://arxiv.org/abs/2302.08113) region-based generation using Stable Diffusion 2.1 model. To get started, draw your masks and type your prompts. More details in the [project page](https://multidiffusion.github.io).
  ''')

  if(is_shared_ui):
    gr.HTML(f'''
           <div style="margin-top:-20px">To skip the queue or try the technique with custom models, you may duplicate the space and associate an A10 GPU to it &nbsp;&nbsp;<a class="duplicate-button" style="display:inline-block" target="_blank" href="https://huggingface.co/spaces/{os.environ['SPACE_ID']}?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></div>
    ''')
  elif(not is_gpu_associated):
      gr.HTML(f'''
           <div>You have succesfully duplicated the Space 🎉, but it is running on CPU - which may break this application. Go to the <a href="https://huggingface.co/spaces/{os.environ['SPACE_ID']}/settings">settings</a> page to associate a GPU to it</div>
      ''')
  with gr.Row():
    with gr.Box(elem_id="main-image"):
      canvas_data = gr.JSON(value={}, visible=False)
      model = gr.Textbox(label="The id of any Hugging Face model in the diffusers format", value="stabilityai/stable-diffusion-2-1-base", visible=False if is_shared_ui else True)
      canvas = gr.HTML(canvas_html)
      aspect = gr.Radio(["square", "horizontal", "vertical"], value="square", label="Aspect Ratio", visible=False if is_shared_ui else True)
      button_run = gr.Button("I've finished my sketch",elem_id="main_button", interactive=True)
      
      prompts = []
      colors = []
      color_row = [None] * MAX_COLORS
      with gr.Column(visible=False) as post_sketch:
        general_prompt = gr.Textbox(label="General Prompt")
        for n in range(MAX_COLORS):
          with gr.Row(visible=False) as color_row[n]:
            with gr.Box(elem_id="color-bg"):
              colors.append(gr.HTML('<div class="color-bg-item" style="background-color: black"></div>'))
            prompts.append(gr.Textbox(label="Prompt for this mask"))
        with gr.Accordion("Advanced options", open=False):
          negative_prompt = gr.Textbox(label="Global negative prompt for all prompts", value="low quality")
          boostrapping = gr.Slider(label="Bootstrapping", minimum=1, maximum=100, value=20, step=1)
          steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
          seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, value=-1, step=1)
        final_run_btn = gr.Button("Generate!")
    
    out_image = gr.Image(label="Result", ).style(width=512,height=512)
  gr.Markdown('''
  ![Examples](https://multidiffusion.github.io/pics/tight.jpg)
  ''')
  #css_height = gr.HTML("<style>#main-image{width: 512px} .fixed-height{height: 512px !important}</style>")
  aspect.change(None, inputs=[aspect], outputs=None, _js = set_canvas_size)
  button_run.click(process_sketch, inputs=[canvas_data, binary_matrixes], outputs=[post_sketch, binary_matrixes, *color_row, *colors], _js=get_js_colors, queue=False)
  final_run_btn.click(process_generation, inputs=[model, binary_matrixes, boostrapping, aspect, steps, seed, general_prompt, negative_prompt, *prompts], outputs=out_image)
  demo.load(None, None, None, _js=load_js)
demo.launch(debug=True, share=True)