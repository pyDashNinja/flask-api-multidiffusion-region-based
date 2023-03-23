## Flask API for Multi Diffusion Region Based

Special thanks to : weizmannscience/multidiffusion-region-based .. All credit goes to him .. I have just added flask API on top.

```
IP : “0.0.0.0:5000/multsd”
```
You will need to do a post request with json object to IP given above.
Json Object Will have these parameters 
```
json object : {
"bg_prompt" : "futuristic sci fi spaceship, star citizen, star atlas, render,",
"bg_nprompt" : "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon,
drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts,
ugly, duplicate, morbid, mutilated, extra fingers, mutated hands",
"fg_prompt" : ["a Husky dog"],
"fg_nprompt" : ["(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch,
cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg
artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands"],
"width" : "512",
"height" : "512",
"ddim_steps" : "35",
"bootstrapping" : "20",
"seed" : "22312",
"masks" :
["base64StringImage"]
}
```

## Note :
* bg_prompts is a string – its the prompts for the background generation
* bg_nprompt is a string – its a negative prompts for the background
* fg_prompt is list of strings – it has multiple strings (prompts) it will be a list of prompts depend on
number of mask you sent it will have that number of strings(prompts) , here in above example it has
one mask so one prompt in the list ["a Husky dog"]
* fg_nprompt it is the negative prompts for fore ground or mask it is also a list of strings it will follow
similar pattern like fg_prompt since fg_prompt has only one prompt string negative prompt for fg will
also have one string in the list like show above in the example
* width it will be an int number
* height it will be an int number
* ddim_steps it will be an int number
* bootstrapping it will be an int number
* seed it will also be an int number
* masks : it will be list of base64string images [“base64StringImage”, “secondBase64Image”] like
that you can send multiple mask with base64images and these are the fg (foreground) so above whenwe defined fg_prompt if we send two mask we will provide two prompts or two strings in the list in
fg_prompt and fg_nprompt as well.
Image Generated through this exampple.


## Setup

```
git clone https://github.com/pyDashNinja/flask-api-multidiffusion-region-based.git
cd flask-api-multidiffusion-region-based
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Result

![alt text](./result/result.png)

## Update 24 MARCH, 2023

Docker Image available for deploying on runpod.io

```
docker pull pydashninja/multi-region
```

Command to run in runpod.io:


```
docker run --gpus all -p 5000:443 multi-region
```


Enjoy!!
