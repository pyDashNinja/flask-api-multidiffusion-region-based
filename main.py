from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
# from flask_cors import CORS
import time
import base64
from io import BytesIO
# from sd import sd
# from cn import cn
import random
import threading
from PIL import Image
# from ldm.models.diffusion.ddim import DDIMSampler
# from cldm.model import create_model, load_state_dict
# from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from region_control import multsd, MultiDiffusion


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# lock = threading.Lock()

@app.before_first_request
def loading():
    device="cuda"
    global sd
    try:
        if next(sd.parameters()).is_cuda:
            pass
        # return jsonify({"message" : "Success"}), 200   
    except:
        sd = MultiDiffusion(device, "2.0")



# @app.route("/putBothModelsInRam", methods=['GET'])
# def putBothModelsInRam():
#     with lock:
#       print("starting")
#       global model
#       global sampler
#       global pipe
#       model = create_model('models/cldm_v15.yaml').cpu()
#       model.load_state_dict(load_state_dict('models/control_sd15_canny.pth', location='cpu'))
#       sampler = DDIMSampler(model)
#       model_id = "stabilityai/stable-diffusion-2"
#       scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
#       pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16 )
#       print("loaded")
#       return jsonify({"message" : "Success"}), 200


# @app.route("/putControlNetInGPU", methods=['GET'])
# def putControlNetInGPU():
#   try:
#     # with lock:
#       print("starting")
#       global model
#       global sampler
#       try:
#         if next(model.parameters()).is_cuda:
#           return jsonify({"message" : "Success"}), 200   
#       except:
#         pass

#       model = create_model('models/cldm_v15.yaml').cpu()
#       model.load_state_dict(load_state_dict('models/control_sd15_canny.pth', location='cpu'))
#       model = model.to("cuda")
#       sampler = DDIMSampler(model)
#       print("loaded")
#       return jsonify({"message" : "Success"}), 200
#   except Exception as E:
#     return jsonify({"message" : "Error of :" + str(E)}), 400

@app.route('/multsd',methods=['POST'])
@cross_origin()
def get_image():
  # with lock:
    try:
    #   # print("hello")
        data = request.get_json()
    #   if data['operation'] == 'sd':
    #     prompt = data['prompt']
    #     nprompt = data['nprompt']
    #     width= int(data['width']) 
    #     height= int(data['height'])
    #     num_inference_steps = int(data['num_inference_steps'])
    #     guidance_scale= float(data['guidance_scale'])
    #     # sd(pipe , prompt, width, height, num_inference_steps, guidance_scale ,neg_prompt):
    #     image = sd(pipe ,prompt, width, height , num_inference_steps , guidance_scale, nprompt)

    #     buffered = BytesIO()
    #     image.save(buffered, format="JPEG") # replace "JPEG" with the format of your image
    #     encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # print(encoded_string)
        # print({'image': encoded_string})
        # return {'image': encoded_string}
        bg_prompt = data['bg_prompt']
        bg_nprompt = data['bg_nprompt']
        fg_prompt = data['fg_prompt']
        fg_nprompt = data['fg_nprompt']
        W = int(data['width'])
        H = int(data['height'])
        ddim_steps = int(data['ddim_steps'])
        bootstrapping = int(data['bootstrapping'])
        seed = int(data['seed'])
        masks = data['masks']
        # print(masks[1])
        masks_in = []
        for mask in masks:
            # print(mask)
            mask = str(mask)
            print("jer")
            base64_bytes = base64.b64decode(mask)
            print("err")
            mask = Image.open(BytesIO(base64_bytes)).convert("L")
            print("jere")
            masks_in.append(mask)
        # print("ehre")
        # print(masks_in)
        image = multsd(sd, masks=masks_in, bg_prompt=bg_prompt, bg_negative=bg_nprompt, fg_prompt=fg_prompt, fg_negative=fg_nprompt, H=H, W=W, steps=ddim_steps, boostrapping=bootstrapping, seed=seed)
        # cn(model, sampler ,prompt, image, image_resolution, ddim_steps, scale, seed, eta, low_threshold, high_threshold,  nprompt):
        # image = multsd(sd, mask=image)    # sd = MultiDiffusion(device, opt.sd_version)


        buffered = BytesIO()
        image.save(buffered, format="JPEG") # replace "JPEG" with the format of your image
        encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # print(encoded_string)
        # print({'image': encoded_string})
        return {'image': encoded_string}

    except Exception as E:
        print(E)
        return jsonify({"message" : "soemthing went wrong" + str(E)}), 400


if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)

# # loading()

# # image = get_image()
# # print(image)

