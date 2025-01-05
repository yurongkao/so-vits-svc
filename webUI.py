import io
import os

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import gradio.processing_utils as gr_pu
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
import logging
import re
import json

import subprocess
import edge_tts
import asyncio
from scipy.io import wavfile
import librosa
import torch
import time
import traceback
from itertools import chain
from utils import mix_model

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None
debug = False

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"

def upload_mix_append_file(files,sfiles):
    try:
        if(sfiles == None):
            file_paths = [file.name for file in files]
        else:
            file_paths = [file.name for file in chain(files,sfiles)]
        p = {file:100 for file in file_paths}
        return file_paths,mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)

def mix_submit_click(js,mode):
    try:
        assert js.lstrip()!=""
        modes = {"Convex Combination":0, "Linear Combination":1}
        mode = modes[mode]
        data = json.loads(js)
        data = list(data.items())
        model_path,mix_rate = zip(*data)
        path = mix_model(model_path,mix_rate,mode)
        return f"Success, file saved in {path}"
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)

def update_mix_info(files):
    try:
        if files == None : return mix_model_output1.update(value="")
        p = {file.name:100 for file in files}
        return mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)

def modelAnalysis(model_path,config_path,cluster_model_path,device,enhance):
    global model
    try:
        device = cuda[device] if "CUDA" in device else device
        model = Svc(model_path.name, config_path.name, device=device if device!="Auto" else None, cluster_model_path = cluster_model_path.name if cluster_model_path != None else "",nsf_hifigan_enhance=enhance)
        spks = list(model.spk2id.keys())
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        msg = f"Model loaded successfully on device {device_name}\n"
        if cluster_model_path is None:
            msg += "Cluster model not loaded\n"
        else:
            msg += f"Cluster model {cluster_model_path.name} loaded successfully\n"
        msg += "Available timbres of the current model:\n"
        for i in spks:
            msg += i + " "
        return sid.update(choices = spks,value=spks[0]), msg
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)

def modelUnload():
    global model
    if model is None:
        return sid.update(choices = [],value=""),"No model to unload!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return sid.update(choices = [],value=""),"Model unloaded successfully!"


def vc_fn(sid, input_audio, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold):
    global model
    try:
        if input_audio is None:
            raise gr.Error("You need to upload audio")
        if model is None:
            raise gr.Error("You need to specify a model")
        sampling_rate, audio = input_audio
        # print(audio.shape,sampling_rate)
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        temp_path = "temp.wav"
        soundfile.write(temp_path, audio, sampling_rate, format="wav")
        _audio = model.slice_inference(temp_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold)
        model.clear_empty()
        os.remove(temp_path)
        try:
            timestamp = str(int(time.time()))
            filename =  timestamp + ".wav"
            output_file = os.path.join("./results", filename)
            soundfile.write(output_file, _audio, model.target_sample, format="wav")
            return f"Inference successful, audio file saved as results/{filename}", (model.target_sample, _audio)
        except Exception as e:
            if debug: traceback.print_exc()
            return f"File saving failed, please save manually", (model.target_sample, _audio)
    except Exception as e:
        if debug: traceback.print_exc()
        raise gr.Error(e)


def tts_func(_text,_rate,_voice):
    voice = "zh-CN-YunxiNeural"#Male
    if ( _voice == "Female" ) : voice = "zh-CN-XiaoyiNeural"
    output_file = _text[0:10]+".wav"
    if _rate>=0:
        ratestr="+{:.0%}".format(_rate)
    elif _rate<0:
        ratestr="{:.0%}".format(_rate)

    p=subprocess.Popen("edge-tts "+
                        " --text "+_text+
                        " --write-media "+output_file+
                        " --voice "+voice+
                        " --rate="+ratestr
                        ,shell=True,
                        stdout=subprocess.PIPE,
                        stdin=subprocess.PIPE)
    p.wait()
    return output_file

def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)

def vc_fn2(sid, input_audio, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,text2tts,tts_rate,tts_voice,f0_predictor,enhancer_adaptive_key,cr_threshold):
    text2tts=text_clear(text2tts)
    output_file=tts_func(text2tts,tts_rate,tts_voice)
    sr2=44100
    wav, sr = librosa.load(output_file)
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=sr2)
    save_path2= text2tts[0:10]+"_44k"+".wav"
    wavfile.write(save_path2,sr2,
                (wav2 * np.iinfo(np.int16).max).astype(np.int16)
                )
    sample_rate, data=gr_pu.audio_from_file(save_path2)
    vc_input=(sample_rate, data)
    a,b=vc_fn(sid, vc_input, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold)
    os.remove(output_file)
    os.remove(save_path2)
    return a,b
def debug_change():
    global debug
    debug = debug_button.value

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue = gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),
) as app:
    with gr.Tabs():
        with gr.TabItem("Inference"):
            gr.Markdown(value="""
                So-vits-svc 4.0 Inference webui
                """)
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=2> Model Settings</font>
                        """)
                    model_path = gr.File(label="Select model file")
                    config_path = gr.File(label="Select configuration file")
                    cluster_model_path = gr.File(label="Select cluster model file (optional if not available)")
                    device = gr.Dropdown(label="Inference device, default is auto select CPU and GPU", choices=["Auto",*cuda.keys(),"CPU"], value="Auto")
                    enhance = gr.Checkbox(label="Use NSF_HIFIGAN enhancement, this option can improve the sound quality of some models with fewer training sets, but it has negative effects on well-trained models, off by default", value=False)
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=3>After selecting all the files on the left side (all file modules show 'download'), click 'Load Model' to parse:</font>
                        """)
                    model_load_button = gr.Button(value="Load model", variant="primary")
                    model_unload_button = gr.Button(value="Unload model", variant="primary")
                    sid = gr.Dropdown(label="Voice tone (speaker)")
                    sid_output = gr.Textbox(label="Output Message")

            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="""
                        <font size=2> Inference Settings</font>
                        """)
                    auto_f0 = gr.Checkbox(label="Automatic f0 prediction, better prediction effect with cluster model, will cause pitch function to fail (only for voice conversion, the song will be extremely off-pitch if checked)", value=False)
                    f0_predictor = gr.Dropdown(label="Select F0 predictor, can choose crepe,pm,dio,harvest,default is pm(Note: crepe is a mean filter for original F0)", choices=["pm","dio","harvest","crepe"], value="pm")
                    vc_transform = gr.Number(label="Pitch change (integer, can be positive or negative, the number of semitones, 12 for an octave increase)", value=0)
                    cluster_ratio = gr.Number(label="Cluster model blending ratio, between 0-1, 0 means not using the cluster. Using the cluster model can increase the similarity of voice tone, but it will cause the bite word to drop (If using, it is recommended to set to around 0.5)", value=0)
                    slice_db = gr.Number(label="Slice threshold", value=-40)
                    noise_scale = gr.Number(label="noise_scale is not recommended to be adjusted, it will affect sound quality, it's a metaphysical parameter", value=0.4)
                with gr.Column():
                    pad_seconds = gr.Number(label="Padding seconds for inference audio. Due to unknown reasons, there will be strange noise at the beginning and end, this will not occur after padding a short silent segment", value=0.5)
                    cl_num = gr.Number(label="Automatic audio slicing, 0 means no slicing, in seconds (s)", value=0)
                    lg_num = gr.Number(label="Crossfade length of the audio slices at both ends. If the speech is incoherent after automatic slicing, adjust this value. If coherent, it is recommended to use the default value of 0. Note that this setting will affect the inference speed, in seconds/s", value=0)
                    lgr_num = gr.Number(label="After automatic audio slicing, it is necessary to discard the beginning and end of each slice. This parameter sets the proportion of cross-fade length to be retained, range 0-1, open on the left and close on the right", value=0.75)
                    enhancer_adaptive_key = gr.Number(label="Adapt the enhancer to a higher range (in semitones)|default is 0", value=0)
                    cr_threshold = gr.Number(label="F0 filtering threshold, only effective when crepe is started. The value range is from 0-1. Lowering this value can reduce the probability of pitch deviation, but it will increase the chance of muteness", value=0.05)
            with gr.Tabs():
                with gr.TabItem("Audio to Audio"):
                    vc_input3 = gr.Audio(label="Select audio")
                    vc_submit = gr.Button("Audio Conversion", variant="primary")
                with gr.TabItem("Text to Audio"):
                    text2tts=gr.Textbox(label="Enter the text to be transcribed here. Note that using this feature is recommended to turn on F0 prediction, otherwise it will be strange")
                    tts_rate = gr.Number(label="tts speed", value=0)
                    tts_voice = gr.Radio(label="Gender",choices=["Male","Female"], value="Male")
                    vc_submit2 = gr.Button("Text Conversion", variant="primary")
            with gr.Row():
                with gr.Column():
                    vc_output1 = gr.Textbox(label="Output Message")
                with gr.Column():
                    vc_output2 = gr.Audio(label="Output Audio", interactive=False)

        with gr.TabItem("Small Tools/Experimental Features"):
            gr.Markdown(value="""
                        <font size=2> So-vits-svc 4.0 Small Tools/Experimental Features</font>
                        """)
            with gr.Tabs():
                with gr.TabItem("Static Voice Line Fusion"):
                    gr.Markdown(value="""
                        <font size=2> Introduction: This feature can merge multiple voice models into one voice model (convex combination or linear combination of multiple model parameters), thus creating voice lines that do not exist in reality 
                                      Note:
                                      1.This feature only supports single speaker models
                                      2.If you forcibly use multi-speaker models, you need to ensure that the number of speakers in multiple models is the same, so you can mix the voices under the same SpeakerID
                                      3.Ensure that the 'model' field in the config.json of all models to be mixed is the same
                                      4.The output mixed model can use any config.json of the model to be synthesized, but the cluster model cannot be used
                                      5.When uploading models in batches, it is best to put the models in a folder and select them all at once
                                      6.The adjustment of the mixing ratio is recommended to be between 0-100, or it can be adjusted to other numbers, but unknown effects will occur in linear combination mode
                                      7.After the mixing is completed, the file will be saved in the root directory of the project, and the filename is output.pth
                                      8.Convex combination mode will perform Softmax on the mixing ratio to make the sum of the mixing ratios equal to 1, while linear combination mode will not
                        </font>
                        """)
                    mix_model_path = gr.Files(label="Select the model files to be mixed")
                    mix_model_upload_button = gr.UploadButton("Select/append model files to be mixed", file_count="multiple", variant="primary")
                    mix_model_output1 = gr.Textbox(
                                            label="Adjust the mixing ratio, unit/%",
                                            interactive = True
                                         )
                    mix_mode = gr.Radio(choices=["Convex Combination", "Linear Combination"], label="Fusion Mode",value="Convex Combination",interactive = True)
                    mix_submit = gr.Button("Start Voice Line Fusion", variant="primary")
                    mix_model_output2 = gr.Textbox(
                                            label="Output Message"
                                         )
                    mix_model_path.change(update_mix_info,[mix_model_path],[mix_model_output1])
                    mix_model_upload_button.upload(upload_mix_append_file, [mix_model_upload_button,mix_model_path], [mix_model_path,mix_model_output1])
                    mix_submit.click(mix_submit_click, [mix_model_output1,mix_mode], [mix_model_output2])
                    
                    
    with gr.Tabs():
        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown(value="""
                    <font size=2> WebUI Settings</font>
                    """)
                debug_button = gr.Checkbox(label="Debug mode, if you need to report BUG to the community, you need to turn it on. After turning on, the console can display specific error prompts", value=debug)
        vc_submit.click(vc_fn, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold], [vc_output1, vc_output2])
        vc_submit2.click(vc_fn2, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,text2tts,tts_rate,tts_voice,f0_predictor,enhancer_adaptive_key,cr_threshold], [vc_output1, vc_output2])
        debug_button.change(debug_change,[],[])
        model_load_button.click(modelAnalysis,[model_path,config_path,cluster_model_path,device,enhance],[sid,sid_output])
        model_unload_button.click(modelUnload,[],[sid,sid_output])
    app.launch()

 