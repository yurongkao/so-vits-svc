import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from itertools import chain
from pathlib import Path

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import librosa
import numpy as np
import soundfile
import torch

from compress_model import removeOptimizer
from edgetts.tts_voices import SUPPORTED_LANGUAGES
from inference.infer_tool import Svc
from utils import mix_model

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None
debug = False

local_model_root = './trained'

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"

def upload_mix_append_file(files,sfiles):
    try:
        if(sfiles is None):
            file_paths = [file.name for file in files]
        else:
            file_paths = [file.name for file in chain(files,sfiles)]
        p = {file:100 for file in file_paths}
        return file_paths,mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
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
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def update_mix_info(files):
    try:
        if files is None :
            return mix_model_output1.update(value="")
        p = {file.name:100 for file in files}
        return mix_model_output1.update(value=json.dumps(p,indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def modelAnalysis(model_path,config_path,cluster_model_path,device,enhance,diff_model_path,diff_config_path,only_diffusion,use_spk_mix,local_model_enabled,local_model_selection):
    global model
    try:
        device = cuda[device] if "CUDA" in device else device
        cluster_filepath = os.path.split(cluster_model_path.name) if cluster_model_path is not None else "no_cluster"
        # get model and config path
        if (local_model_enabled):
            # local path
            model_path = glob.glob(os.path.join(local_model_selection, '*.pth'))[0]
            config_path = glob.glob(os.path.join(local_model_selection, '*.json'))[0]
        else:
            # upload from webpage
            model_path = model_path.name
            config_path = config_path.name
        fr = ".pkl" in cluster_filepath[1]
        model = Svc(model_path,
                config_path,
                device=device if device != "Auto" else None,
                cluster_model_path = cluster_model_path.name if cluster_model_path is not None else "",
                nsf_hifigan_enhance=enhance,
                diffusion_model_path = diff_model_path.name if diff_model_path is not None else "",
                diffusion_config_path = diff_config_path.name if diff_config_path is not None else "",
                shallow_diffusion = True if diff_model_path is not None else False,
                only_diffusion = only_diffusion,
                spk_mix_enable = use_spk_mix,
                feature_retrieval = fr
                )
        spks = list(model.spk2id.keys())
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        msg = f"Model loaded successfully on device {device_name}\n"
        if cluster_model_path is None:
<<<<<<< HEAD
            msg += "Cluster model not loaded\n"
        else:
            msg += f"Cluster model {cluster_model_path.name} loaded successfully\n"
        msg += "Available timbres of the current model:\n"
=======
            msg += "未加载聚类模型或特征检索模型\n"
        elif fr:
            msg += f"特征检索模型{cluster_filepath[1]}加载成功\n"
        else:
            msg += f"聚类模型{cluster_filepath[1]}加载成功\n"
        if diff_model_path is None:
            msg += "未加载扩散模型\n"
        else:
            msg += f"扩散模型{diff_model_path.name}加载成功\n"
        msg += "当前模型的可用音色：\n"
>>>>>>> 730930d337d171479eadf305f96cbed4bb393e77
        for i in spks:
            msg += i + " "
        return sid.update(choices = spks,value=spks[0]), msg
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

def modelUnload():
    global model
    if model is None:
        return sid.update(choices = [],value=""),"No model to unload!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
<<<<<<< HEAD
        return sid.update(choices = [],value=""),"Model unloaded successfully!"
=======
        return sid.update(choices = [],value=""),"模型卸载完毕!"
    
def vc_infer(output_format, sid, audio_path, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model
    _audio = model.slice_inference(
        audio_path,
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment
    )  
    model.clear_empty()
    #构建保存文件的路径，并保存到results文件夹内
    str(int(time.time()))
    if not os.path.exists("results"):
        os.makedirs("results")
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff"
>>>>>>> 730930d337d171479eadf305f96cbed4bb393e77

    if model.only_diffusion:
        isdiffusion = "diff"
    
    output_file_name = 'result_'+truncated_basename+f'_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
    output_file = os.path.join("results", output_file_name)
    soundfile.write(output_file, _audio, model.target_sample, format=output_format)
    return output_file

def vc_fn(sid, input_audio, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    global model
    try:
        if input_audio is None:
<<<<<<< HEAD
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
=======
            return "You need to upload an audio", None
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        #print(input_audio)    
        audio, sampling_rate = soundfile.read(input_audio)
        #print(audio.shape,sampling_rate)
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        #print(audio.dtype)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        # 未知原因Gradio上传的filepath会有一个奇怪的固定后缀，这里去掉
        truncated_basename = Path(input_audio).stem[:-6]
        processed_audio = os.path.join("raw", f"{truncated_basename}.wav")
        soundfile.write(processed_audio, audio, sampling_rate, format="wav")
        output_file = vc_infer(output_format, sid, processed_audio, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)

        return "Success", output_file
>>>>>>> 730930d337d171479eadf305f96cbed4bb393e77
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

<<<<<<< HEAD

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
=======
def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)

def vc_fn2(_text, _lang, _gender, _rate, _volume, sid, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold, k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    global model
    try:
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        _rate = f"+{int(_rate*100)}%" if _rate >= 0 else f"{int(_rate*100)}%"
        _volume = f"+{int(_volume*100)}%" if _volume >= 0 else f"{int(_volume*100)}%"
        if _lang == "Auto":
            _gender = "Male" if _gender == "男" else "Female"
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume, _gender])
        else:
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume])
        target_sr = 44100
        y, sr = librosa.load("tts.wav")
        resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        soundfile.write("tts.wav", resampled_y, target_sr, subtype = "PCM_16")
        input_audio = "tts.wav"
        #audio, _ = soundfile.read(input_audio)
        output_file_path = vc_infer(output_format, sid, input_audio, "tts", vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
        os.remove("tts.wav")
        return "Success", output_file_path
    except Exception as e:
        if debug: traceback.print_exc()  # noqa: E701
        raise gr.Error(e)

def model_compression(_model):
    if _model == "":
        return "请先选择要压缩的模型"
    else:
        model_path = os.path.split(_model.name)
        filename, extension = os.path.splitext(model_path[1])
        output_model_name = f"{filename}_compressed{extension}"
        output_path = os.path.join(os.getcwd(), output_model_name)
        removeOptimizer(_model.name, output_path)
        return f"模型已成功被保存在了{output_path}"

def scan_local_models():
    res = []
    candidates = glob.glob(os.path.join(local_model_root, '**', '*.json'), recursive=True)
    candidates = set([os.path.dirname(c) for c in candidates])
    for candidate in candidates:
        jsons = glob.glob(os.path.join(candidate, '*.json'))
        pths = glob.glob(os.path.join(candidate, '*.pth'))
        if (len(jsons) == 1 and len(pths) == 1):
            # must contain exactly one json and one pth file
            res.append(candidate)
    return res

def local_model_refresh_fn():
    choices = scan_local_models()
    return gr.Dropdown.update(choices=choices)

>>>>>>> 730930d337d171479eadf305f96cbed4bb393e77
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
<<<<<<< HEAD
                    model_path = gr.File(label="Select model file")
                    config_path = gr.File(label="Select configuration file")
                    cluster_model_path = gr.File(label="Select cluster model file (optional if not available)")
                    device = gr.Dropdown(label="Inference device, default is auto select CPU and GPU", choices=["Auto",*cuda.keys(),"CPU"], value="Auto")
                    enhance = gr.Checkbox(label="Use NSF_HIFIGAN enhancement, this option can improve the sound quality of some models with fewer training sets, but it has negative effects on well-trained models, off by default", value=False)
=======
                    with gr.Tabs():
                        # invisible checkbox that tracks tab status
                        local_model_enabled = gr.Checkbox(value=False, visible=False)
                        with gr.TabItem('上传') as local_model_tab_upload:
                            with gr.Row():
                                model_path = gr.File(label="选择模型文件")
                                config_path = gr.File(label="选择配置文件")
                        with gr.TabItem('本地') as local_model_tab_local:
                            gr.Markdown(f'模型应当放置于{local_model_root}文件夹下')
                            local_model_refresh_btn = gr.Button('刷新本地模型列表')
                            local_model_selection = gr.Dropdown(label='选择模型文件夹', choices=[], interactive=True)
                    with gr.Row():
                        diff_model_path = gr.File(label="选择扩散模型文件")
                        diff_config_path = gr.File(label="选择扩散模型配置文件")
                    cluster_model_path = gr.File(label="选择聚类模型或特征检索文件（没有可以不选）")
                    device = gr.Dropdown(label="推理设备，默认为自动选择CPU和GPU", choices=["Auto",*cuda.keys(),"cpu"], value="Auto")
                    enhance = gr.Checkbox(label="是否使用NSF_HIFIGAN增强,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭", value=False)
                    only_diffusion = gr.Checkbox(label="是否使用全扩散推理，开启后将不使用So-VITS模型，仅使用扩散模型进行完整扩散推理，默认关闭", value=False)
>>>>>>> 730930d337d171479eadf305f96cbed4bb393e77
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
<<<<<<< HEAD
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
=======
                    auto_f0 = gr.Checkbox(label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声勾选此项会究极跑调）", value=False)
                    f0_predictor = gr.Dropdown(label="选择F0预测器,可选择crepe,pm,dio,harvest,rmvpe,默认为pm(注意：crepe为原F0使用均值滤波器)", choices=["pm","dio","harvest","crepe","rmvpe"], value="pm")
                    vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
                    cluster_ratio = gr.Number(label="聚类模型/特征检索混合比例，0-1之间，0即不启用聚类/特征检索。使用聚类/特征检索能提升音色相似度，但会导致咬字下降（如果使用建议0.5左右）", value=0)
                    slice_db = gr.Number(label="切片阈值", value=-40)
                    output_format = gr.Radio(label="音频输出格式", choices=["wav", "flac", "mp3"], value = "wav")
                    noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
                    k_step = gr.Slider(label="浅扩散步数，只有使用了扩散模型才有效，步数越大越接近扩散模型的结果", value=100, minimum = 1, maximum = 1000)
                with gr.Column():
                    pad_seconds = gr.Number(label="推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现", value=0.5)
                    cl_num = gr.Number(label="音频自动切片，0为不切片，单位为秒(s)", value=0)
                    lg_num = gr.Number(label="两端音频切片的交叉淡入长度，如果自动切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，注意，该设置会影响推理速度，单位为秒/s", value=0)
                    lgr_num = gr.Number(label="自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭", value=0.75)
                    enhancer_adaptive_key = gr.Number(label="使增强器适应更高的音域(单位为半音数)|默认为0", value=0)
                    cr_threshold = gr.Number(label="F0过滤阈值，只有启动crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音", value=0.05)
                    loudness_envelope_adjustment = gr.Number(label="输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络", value = 0)
                    second_encoding = gr.Checkbox(label = "二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，效果时好时差，默认关闭", value=False)
                    use_spk_mix = gr.Checkbox(label = "动态声线融合", value = False, interactive = False)
            with gr.Tabs():
                with gr.TabItem("音频转音频"):
                    vc_input3 = gr.Audio(label="选择音频", type="filepath")
                    vc_submit = gr.Button("音频转换", variant="primary")
                with gr.TabItem("文字转音频"):
                    text2tts=gr.Textbox(label="在此输入要转译的文字。注意，使用该功能建议打开F0预测，不然会很怪")
                    with gr.Row():
                        tts_gender = gr.Radio(label = "说话人性别", choices = ["男","女"], value = "男")
                        tts_lang = gr.Dropdown(label = "选择语言，Auto为根据输入文字自动识别", choices=SUPPORTED_LANGUAGES, value = "Auto")
                        tts_rate = gr.Slider(label = "TTS语音变速（倍速相对值）", minimum = -1, maximum = 3, value = 0, step = 0.1)
                        tts_volume = gr.Slider(label = "TTS语音音量（相对值）", minimum = -1, maximum = 1.5, value = 0, step = 0.1)
                    vc_submit2 = gr.Button("文字转换", variant="primary")
>>>>>>> 730930d337d171479eadf305f96cbed4bb393e77
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
<<<<<<< HEAD
                    mix_model_path = gr.Files(label="Select the model files to be mixed")
                    mix_model_upload_button = gr.UploadButton("Select/append model files to be mixed", file_count="multiple", variant="primary")
=======
                    mix_model_path = gr.Files(label="选择需要混合模型文件")
                    mix_model_upload_button = gr.UploadButton("选择/追加需要混合模型文件", file_count="multiple")
>>>>>>> 730930d337d171479eadf305f96cbed4bb393e77
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
                
                with gr.TabItem("模型压缩工具"):
                    gr.Markdown(value="""
                        该工具可以实现对模型的体积压缩，在**不影响模型推理功能**的情况下，将原本约600M的So-VITS模型压缩至约200M, 大大减少了硬盘的压力。
                        **注意：压缩后的模型将无法继续训练，请在确认封炉后再压缩。**
                    """)
                    model_to_compress = gr.File(label="模型上传")
                    compress_model_btn = gr.Button("压缩模型", variant="primary")
                    compress_model_output = gr.Textbox(label="输出信息", value="")

                    compress_model_btn.click(model_compression, [model_to_compress], [compress_model_output])
                    
                    
    with gr.Tabs():
        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown(value="""
                    <font size=2> WebUI Settings</font>
                    """)
<<<<<<< HEAD
                debug_button = gr.Checkbox(label="Debug mode, if you need to report BUG to the community, you need to turn it on. After turning on, the console can display specific error prompts", value=debug)
        vc_submit.click(vc_fn, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold], [vc_output1, vc_output2])
        vc_submit2.click(vc_fn2, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,text2tts,tts_rate,tts_voice,f0_predictor,enhancer_adaptive_key,cr_threshold], [vc_output1, vc_output2])
=======
                debug_button = gr.Checkbox(label="Debug模式，如果向社区反馈BUG需要打开，打开后控制台可以显示具体错误提示", value=debug)
        # refresh local model list
        local_model_refresh_btn.click(local_model_refresh_fn, outputs=local_model_selection)
        # set local enabled/disabled on tab switch
        local_model_tab_upload.select(lambda: False, outputs=local_model_enabled)
        local_model_tab_local.select(lambda: True, outputs=local_model_enabled)
        
        vc_submit.click(vc_fn, [sid, vc_input3, output_format, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment], [vc_output1, vc_output2])
        vc_submit2.click(vc_fn2, [text2tts, tts_lang, tts_gender, tts_rate, tts_volume, sid, output_format, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment], [vc_output1, vc_output2])

>>>>>>> 730930d337d171479eadf305f96cbed4bb393e77
        debug_button.change(debug_change,[],[])
        model_load_button.click(modelAnalysis,[model_path,config_path,cluster_model_path,device,enhance,diff_model_path,diff_config_path,only_diffusion,use_spk_mix,local_model_enabled,local_model_selection],[sid,sid_output])
        model_unload_button.click(modelUnload,[],[sid,sid_output])
    os.system("start http://127.0.0.1:7860")
    app.launch()

<<<<<<< HEAD
 
=======

 
>>>>>>> 730930d337d171479eadf305f96cbed4bb393e77
