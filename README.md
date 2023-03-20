## Flask API for Multi Diffusion Region Based

Special thanks to : weizmannscience/multidiffusion-region-based .. All credit goes to him .. I have just added flask API on top.

``
IP : “0.0.0.0:5000/multsd”
``
You will need to do a post request with json object to IP given above.
Json Object Will have these parameters 
``
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
``

#### Note :
1 - bg_prompts is a string – its the prompts for the background generation
2 – bg_nprompt is a string – its a negative prompts for the background
3 – fg_prompt is list of strings – it has multiple strings (prompts) it will be a list of prompts depend on
number of mask you sent it will have that number of strings(prompts) , here in above example it has
one mask so one prompt in the list ["a Husky dog"]
4 - fg_nprompt it is the negative prompts for fore ground or mask it is also a list of strings it will follow
similar pattern like fg_prompt since fg_prompt has only one prompt string negative prompt for fg will
also have one string in the list like show above in the example
5 – width it will be an int number
6 – height it will be an int number
7 – ddim_steps it will be an int number
8 – bootstrapping it will be an int number
9 – seed it will also be an int number
10 – masks : it will be list of base64string images [“base64StringImage”, “secondBase64Image”] like
that you can send multiple mask with base64images and these are the fg (foreground) so above whenwe defined fg_prompt if we send two mask we will provide two prompts or two strings in the list in
fg_prompt and fg_nprompt as well.
Image Generated through this exampple.


## Setup

``
git clone https://github.com/pyDashNinja/flask-api-multidiffusion-region-based.git
cd flask-api-multidiffusion-region-based
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
``

Enjoy!!
