from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
from io import BytesIO
from PIL import Image
from multidiffusion import multsd, MultiDiffusion


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


@app.route('/multsd',methods=['POST'])
@cross_origin()
def get_image():
    try:
        data = request.get_json()
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
        
        image = multsd(sd, masks=masks_in, bg_prompt=bg_prompt, bg_negative=bg_nprompt, fg_prompt=fg_prompt, fg_negative=fg_nprompt, H=H, W=W, steps=ddim_steps, boostrapping=bootstrapping, seed=seed)


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

