import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel
from utils.utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor
from utils.models import CogAgentModel, CogVLMModel
from PIL import Image
import time
from torch.utils.data import Dataset, DataLoader
import cv2
import openai
import json

def initialization():
    openai.api_key = 'XXXXXXX'

def format_message(message, role="user"):
    return {
        "role": role,
        "content": message,
    }

def load_data(videopath='XXXXX/data/videos', frame_rate=2):
    video_paths = list_video_files(videopath)
    frame_dataset = VideoFrameDataset(video_paths, frame_rate=frame_rate)
    dataloader = DataLoader(frame_dataset, shuffle=False)
    return dataloader

def load_data_fromjson(videopath, frame_rate):
    frame_dataset = VideoFrameDataset(videopath, frame_rate=frame_rate)
    dataloader = DataLoader(frame_dataset, shuffle=False)
    return dataloader

def is_video_file(file_name):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm']
    _, file_extension = os.path.splitext(file_name)
    return file_extension.lower() in video_extensions

def list_video_files(videopath):
    video_files = []
    # Ensure the provided path is a directory
    if not os.path.isdir(videopath):
        print(f"Error: '{videopath}' is not a valid directory.")
        return video_files

    # Iterate over all files in the directory
    for file_name in os.listdir(videopath):
        file_path = os.path.join(videopath, file_name)
        if os.path.isfile(file_path) and is_video_file(file_name):
            video_files.append(file_path)

    return video_files

class VideoFrameDataset(Dataset):
    def __init__(self, video_paths, frame_rate):
        self.video_paths = video_paths
        self.frame_rate = frame_rate
        self.frames_list, self.video_names = self.extract_frames_from_videos()

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        return {"video": self.frames_list[idx], "videoname": self.video_names[idx]}

    def extract_frames_from_videos(self):
        frames_list = []
        video_names = []
        for video_path in self.video_paths:
            frames = []
            video_capture = cv2.VideoCapture(video_path)
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_interval = int(round(fps / self.frame_rate))
            for i in range(0, total_frames, frame_interval):
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
                success, frame = video_capture.read()
                if not success:
                    continue
                # Convert frame from BGR to RGB (OpenCV uses BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            video_capture.release()
            frames_list.append(frames)
            video_names.append(video_path)        
        return frames_list, video_names

def readoutput(outpath, root='~', captionroot='~'):
    with open(outpath, 'r') as f:
        interaction_captions = json.load(f)
    video_paths = interaction_captions.keys()
    return interaction_captions, video_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--chinese", action='store_true', help='Chinese interface')
    parser.add_argument("--version", type=str, default="chat", choices=['chat', 'vqa', 'chat_old', 'base'], help='version of language process. if there is \"text_processor_version\" in model_config.json, this option will be overwritten')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')

    parser.add_argument("--from_pretrained", type=str, default="cogagent-chat", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--stream_chat", action="store_true")
    parser.add_argument("--runn", type=int, default=0)
    parser.add_argument("--startindex", type=int, default=-1)
    parser.add_argument("--endindex", type=int, default=10000)
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args = parser.parse_args()

    initialization()

    # load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=rank,
        rank=rank,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cpu' if args.quant else 'cuda',
        **vars(args)
    ), overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else args.version
    print("[Language processor version]:", language_processor_version)
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=language_processor_version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None
    
    if args.quant:
        quantize(model, args.quant)
        if torch.cuda.is_available():
            model = model.cuda()

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)
    
    rootdir = 'XXXXX/data/'

    vila_outpath = os.path.join(rootdir, 'outputs/vila.json')
    gpt4_outpath = os.path.join(rootdir, 'outputs/gpt4.json')
    gemini_outpath = os.path.join(rootdir, 'outputs/gemini.json')

    vila_captions, videopaths = readoutput(vila_outpath)
    gemini_captions, _ = readoutput(gemini_outpath)
    gpt4_captions, _ = readoutput(gpt4_outpath)

    dataloader = load_data_fromjson(videopaths, frame_rate=1)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i <= args.startindex:
                print(i)
                continue
            if i > args.endindex:
                continue
            if batch["videoname"][0] != videopaths[i]:
                print('ERROR: The video does not match the name!')
                import pdb; pdb.set_trace()
            video_path = videopaths[i]
            video_len = len(batch["video"])
            videosum = ''
            history = None
            for j in range(video_len):
                # Convert to PIL Image format
                pil_image = Image.fromarray(batch["video"][j][0].numpy())
                input_image = pil_image
                query = "Please describe the visual and narrative elements of the video in detail, particularly the motion behavior."
                try:
                    response, _, _ = chat(
                        input_image,
                        model,
                        text_processor_infer,
                        image_processor,
                        query,
                        history=history,
                        cross_img_processor=cross_image_processor,
                        image=None,
                        max_length=args.max_length,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        invalid_slices=text_processor_infer.invalid_slices,
                        args=args
                        )
                except Exception as e:
                    print(e)
                    break
                if response.startswith('<s>'):
                    history = None
                    continue
                videosum += 'No.{} frame:{}\n'.format(j, response)
                if j == video_len // 2:
                    cogagent_caption = response
            query = 'summarize all the description to describe a video with accurate temporal information: --> {}'.format(videosum)
    
            cogvideo_caption = openai.ChatCompletion.create(model="gpt-4", 
                                        messages=[format_message(query)], 
                                        temperature=0.0,)
            cogvideo_caption = cogvideo_caption["choices"][0]["message"]["content"]
            vila_caption = vila_captions[video_path]['output']
            gemini_caption = gemini_captions[video_path]['output']
            gpt4_caption = gpt4_captions[video_path]['output']

            query = 'Please summarize on the visual and narrative elements of the video in detail from descriptions from Image Model A ({}) and Image Model B ({}) and descriptions from Video Model A ({}) and Video Model B ({}).'.format(cogvideo_caption, gpt4_caption, vila_caption, gemini_caption)
            wolf_caption = openai.ChatCompletion.create(model="gpt-4", 
                                        messages=[format_message(query)], 
                                        temperature=0.0,)
            wolf_caption = wolf_caption["choices"][0]["message"]["content"]

if __name__ == "__main__":
    main()
