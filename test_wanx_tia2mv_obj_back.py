print('start test')
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings
import pyloudnorm as pyln
import jieba
import soundfile as sf
import librosa
warnings.filterwarnings('ignore')
import numpy as np
import torch, random
import torch.distributed as dist
from PIL import Image
import pandas as pd
import librosa
import re
import json
import cv2
import wan
from einops import rearrange
from utils.audio_analysis.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video, cache_image, str2bool
from utils.img_utils import get_image_from_path,_center_crop_to_aspect_ratio,process_images_final,resize_images,resize_short_side,get_first_frame_from_video,get_all_frames_from_video,read_obj_tensor_from_path
def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio
def align_floor_to(value, alignment):
    return int(math.floor(value / alignment) * alignment)
def align_ceil_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)
def custom_init(device, wav2vec):    
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder


def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25 # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    return audio_emb
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="flf2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--transformer_dir",
        type=str,
        default=None,
        help="The path to the transformer directory.")
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="The path to the lora directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--back_append_frame",
        type=int,
        default=1)
    parser.add_argument(
        "--start",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--short_side",
        type=int,
        default=512,
        help="The short side of the image or video.")
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--text_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--audio_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--bad_cfg",
        action="store_true",
        default=False,
        help="Whether to use bad CFG.")
    parser.add_argument(
        "--all_text",
        action="store_true",
        default=False,
        help="Whether to use bad CFG.")
    parser.add_argument(
        "--bad_thres",
        type=float,
        default=0,
        help="The threshold for bad CFG.")
    parser.add_argument(
        "--three_cfg",
        action="store_true",
        default=False,
        help="Whether to use three CFG.")
    parser.add_argument(
        "--caption",
        type=str,
        default="short",
        help="The caption type.")
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
        help="The path to the test data.")
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--load_from_merged_model",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")

    args = parser.parse_args()
    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device_id = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = WAN_CONFIGS['ti2v-5B']
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]
    assert args.back_append_frame in [2,1], f"`{args.back_append_frame=}` should be 2 or 1."
    logging.info("Creating WanI2V pipeline.")
    wan_a2v = wan.WanTIA2MVRefBackIDPrefix(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        transformer_dir=args.transformer_dir,
        device_id=device_id,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
        load_from_merged_model=args.load_from_merged_model,
        short_side=args.short_side,
        back_append_frame=args.back_append_frame
    )

    device = torch.device(f"cuda:{device_id}")
    wav2vec_feature_extractor, audio_encoder= custom_init(device, args.wav2vec_dir)

    test_data_path =  args.test_data_path
    with open(test_data_path, 'r') as f:
        test_dataset = json.load(f)
    print(len(test_dataset),args.start, args.end)
    for idx in range(len(test_dataset)):
        if args.start <= idx <= args.end:
            batch = test_dataset[idx]
        else:
            continue
        save_file = os.path.join(args.save_path, batch["video_id"] + '.mp4')
        if os.path.exists(save_file):
            print('skip')
            continue
        mode = args.mode
        negative_prompt = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        if batch.get('mp4_img',None) is not None:
            img = get_image_from_path(batch['mp4_img'])
        else:
            img = get_first_frame_from_video(batch['mp4_path'])
        if batch.get('dwpose_img',None) is not None:
            dw_img = get_image_from_path(batch['dwpose_img'])
        else:
            dw_img = get_first_frame_from_video(batch['dwpose_mp4'])
        
        if batch.get('dwpose_mp4',None) is not None:
            dw_seqs = get_all_frames_from_video(batch['dwpose_mp4'])
            if batch.get('clips',None) is not None:
                clips = batch['clips']
                dw_seqs = dw_seqs[clips[0]:clips[1]]
        else:
            dwpose_frame_num = args.frame_num
            dw_seqs = None
        w, h = img.size
        img = resize_short_side(img, args.short_side)
        small_img = resize_short_side(img, 256)
        dw_img = resize_short_side(dw_img, 256)
        if args.short_side == 512:
            _, small_img = resize_images(img, small_img)
            img, dw_img = resize_images(img, dw_img)
        else:
            _, small_img = process_images_final(img, small_img)
            img, dw_img = process_images_final(img, dw_img)
        dw_w, dw_h = dw_img.size
        frame_num = args.frame_num
        if small_img.size != dw_img.size:
            small_img = img.resize(dw_img.size, Image.LANCZOS)
        if dw_seqs is not None:
            dw_seqs = [dw.resize(dw_img.size, Image.LANCZOS) for dw in dw_seqs]
            dwpose_len = len(dw_seqs)
            dwpose_frame_num = (dwpose_len - 1) // 4 * 4 + 1
        
        if batch.get('audio',None) is not None:
            audio_path = batch["audio"]
            audio_input, sampling_rate = librosa.load(audio_path, sr=16000)
            audio_frames_clip = loudness_norm(audio_input, sampling_rate)
            audio_frame_len = int(len(audio_frames_clip) / sampling_rate * 25)
            audio_frame_num = (audio_frame_len - 1) // 4 * 4 + 1
        elif batch.get('speech_path',None) is not None:
            audio_path = batch["speech_path"]
            if not os.path.exists(audio_path):
                print('error', audio_path)
                continue
            audio_input, sampling_rate = librosa.load(audio_path, sr=16000)
            audio_frames_clip = loudness_norm(audio_input, sampling_rate)
            audio_frame_len = int(len(audio_frames_clip) / sampling_rate * 25)
            audio_frame_num = (audio_frame_len - 1) // 4 * 4 + 1
        elif batch.get('spoken_text_en_interactive_path',None) is not None:
            voice_list = batch['voice_list']
            voice = random.choice(voice_list)
            audio_path = os.path.join(batch['spoken_text_en_interactive_path'],voice+'.wav')
            if not os.path.exists(audio_path):
                print('error', audio_path)
                continue
            audio_input, sampling_rate = librosa.load(audio_path, sr=16000)
            audio_frames_clip = loudness_norm(audio_input, sampling_rate)
            audio_frame_len = int(len(audio_frames_clip) / sampling_rate * 25)
            audio_frame_num = (audio_frame_len - 1) // 4 * 4 + 1
        else:
            audio_frames_clip = np.zeros((int(dwpose_frame_num / 25 * 16000)))
            audio_frame_num = dwpose_frame_num
            sampling_rate = 16000
        if dw_seqs is not None:
            dw_seqs = dw_seqs[:frame_num]
            if audio_frame_num > dwpose_frame_num:
                padding_dwpose = dw_seqs[-1]
                dw_seqs = dw_seqs + [padding_dwpose] * (audio_frame_num - dwpose_frame_num)
                dwpose_frame_num = audio_frame_num
            if audio_frame_num < dwpose_frame_num:
                padding_audio = np.zeros((int(dwpose_frame_num / 25 * 16000)))
                audio_frames_clip = np.concatenate([audio_frames_clip, padding_audio], axis=0)
                audio_frame_num = dwpose_frame_num
        if dw_seqs is None and mode == 'ap2v':
            dw_seqs = [dw_img] * audio_frame_num
        frame_num = min(args.frame_num,min(audio_frame_num,dwpose_frame_num))
        audio_frames_clip = audio_frames_clip[:int(frame_num * sampling_rate / 25)]
        zero_audio_embedding = get_embedding(np.zeros_like(audio_frames_clip), wav2vec_feature_extractor, audio_encoder, device=device) 
        audio_embedding = get_embedding(audio_frames_clip, wav2vec_feature_extractor, audio_encoder, device=device)

        obj_img_path = batch.get('obj_img',None)
        obj_img = read_obj_tensor_from_path(obj_img_path,target_size=img.size)
        caption_prompt_list = []
        clear_state = " clear hands and face, objects are clear and stable. human movements are slow and steady, with a strong sense of reality."
        if args.back_append_frame == 1: 
            segments = re.findall(r'\(.*?\)', batch["structured_prompt"])
            if args.all_text:
                first_three_segments = segments
            else:
                first_three_segments = segments[:3]
            print(first_three_segments)
            result_string = "".join(first_three_segments)
            caption_prompt = batch["prompt"] + clear_state + result_string.lower().replace(')(',') (')
        else:
            for i in range(len(batch["structured_prompt"])):
                segments = re.findall(r'\(.*?\)', batch["structured_prompt"][i])
                if args.all_text:
                    first_three_segments = segments
                else:
                    first_three_segments = segments[:3]
                print(first_three_segments)
                result_string = "".join(first_three_segments)
                caption_prompt = batch["prompt"][i] + clear_state + result_string.lower().replace(')(',') (')
                caption_prompt_list.append(caption_prompt)
        if mode in ['a2v','a2mv','mv','i2v']:
            dw_seqs = None

        logging.info("Generating video ...")
        gen_kwargs = dict(
            id_img=img,
            id_small_img=small_img,
            pose_sequence=dw_seqs,
            mode=mode,
            audio_embedding=audio_embedding,
            zero_audio_embedding=zero_audio_embedding,
            n_prompt=negative_prompt,
            frame_num=frame_num,
            shift=args.sample_shift,
            sampling_steps=args.sample_steps,
            text_guide_scale=args.text_guide_scale,
            audio_guide_scale=args.audio_guide_scale,
            seed=args.base_seed,
            bad_cfg=args.bad_cfg,
            three_cfg=args.three_cfg,
            bad_thres=args.bad_thres,
            offload_model=args.offload_model,
            motion_frame=25,
            max_frames_num=frame_num,
        )
        if args.back_append_frame == 1:
            video, _ = wan_a2v.generate(
                caption_prompt, img, dw_img, obj_img, small_img, **gen_kwargs
            )
        else:
            video, _ = wan_a2v.generate_long(
                caption_prompt_list, img, dw_img, obj_img, small_img, **gen_kwargs
            )
        print('use of bad_cfg:',args.bad_cfg)
        if rank == 0:
            from wan.utils.multitalk_utils import save_composite_video_with_audio
            os.makedirs(args.save_path, exist_ok=True)
            img.save(os.path.join(args.save_path, batch["video_id"] + '.jpg'))
            save_file = os.path.join(args.save_path, batch["video_id"] + '.mp4')
            logging.info(f"Saving generated video to {save_file}")
            chinese_font_path = "./DroidSansFallback.ttf"
            save_composite_video_with_audio(video, save_file,text=batch["prompt_zh"],font_path=chinese_font_path)




    logging.info("Finished.")

import time
if __name__ == "__main__":
    args = _parse_args()

    time0 = time.time()
    generate(args)
    total_time = time.time() - time0
    print(f"Total time: {total_time:.2f} seconds")
    