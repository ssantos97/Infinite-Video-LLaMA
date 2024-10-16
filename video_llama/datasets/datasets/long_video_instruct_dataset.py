import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import pathlib
import json
import numpy as np
import io
import tarfile
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import copy
from video_llama.processors import transforms_video,AlproVideoTrainProcessor
from torchvision import transforms
from video_llama.processors.video_processor import ToTHWC,ToUint8,load_video
from video_llama.conversation.conversation_video import Conversation,SeparatorStyle

DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
video_conversation = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
llama_v2_video_conversation = Conversation(
    system=" ",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)
IGNORE_INDEX = -100

class Long_Video_Instruct_Dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_root, subt_root, num_video_query_token=32,tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/',data_type = 'video', model_type='vicuna'):
        """
        vis_root (string): Root directory of Llava images (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        data_path = pathlib.Path(ann_root)
        with data_path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)

        self.num_video_query_token = num_video_query_token
        self.vis_root = vis_root
        self.subtitle_path = subt_root
        self.resize_size = 224
        self.num_frm = 8
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]

        self.transform = AlproVideoTrainProcessor(
            image_size=self.resize_size, n_frms = self.num_frm
        ).transform
        self.data_type = data_type
        self.model_type = model_type

    def _get_video_path(self, sample):
        rel_video_fp = sample['video']
        full_video_fp = os.path.join(self.vis_root,  rel_video_fp)
        return full_video_fp

    def load_video(self, video_path, height, width):
        video_file = tarfile.open(video_path, 'r')
        image_file = [x for x in video_file.getmembers() if '.jpg' in x.name]
        image_data = [video_file.extractfile(x).read() for x in image_file]
        image_data = [torch.from_numpy(np.array(Image.open(io.BytesIO(x)).resize(height, width))).permute(2, 0, 1).unsqueeze(1) for x in image_data]

        file_name = [x.name for x in image_file]
        indexed_list = list(enumerate(file_name))
        indexed_list = sorted(indexed_list, key=lambda x: x[1])
        image_data = [image_data[x] for x, _ in indexed_list]
        msg = f"The video contains key {len(image_data)} frames sampled from all movie actions."
        return torch.stack(image_data, dim=1), msg
    
    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]

                video_path = self._get_video_path(sample)
                conversation_list = sample['conversations']
                id = sample["video"].replace("/", "").replace(".tar", "")
                id = self.subtitle_path + id + ".srt"
                video, msg = load_video(
                    video_path=video_path,
                    height=self.resize_size,
                    width=self.resize_size,
                )
                video = self.transform(video)
                if 'cn' in self.data_type:
                    msg = ""
                # 添加视频<DEFAULT_IMAGE_PATCH_TOKEN>,以及msg到convsation list 0
                new_sources = preprocess_multimodal(copy.deepcopy(conversation_list), None, cur_token_len=self.num_video_query_token, msg=msg, id=id)
                
                if self.model_type =='vicuna':
                    data_dict = preprocess(
                        new_sources,
                        self.tokenizer)
                elif self.model_type =='llama_v2':
                    data_dict = preprocess_for_llama_v2(
                        new_sources,
                        self.tokenizer)
                else:
                    print('not support')
                    raise('not support')
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])
                # image exist in the data
                data_dict['image'] = video
            except:
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "text_input": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "type":'video',
        }

    def __len__(self):
        return len(self.annotation)

    def collater(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("text_input", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        batch['conv_type'] = 'multi'
        return batch

def load_subtitles(file_path):
    def check(subtitle):
        # We only check if subtitle text is not None
        return subtitle.get('text', None) is not None 

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()

    subtitle = ''
    
    for line in lines:
        line = line.replace('\x00', '').strip()
        line = line.replace('...', '').strip()
        if not line:
            continue
        
        if line.isdigit():
            # Skip if we reach a digit line (subtitle number)
            continue
        elif ' --> ' in line:
            # Skip start and end times
            continue
        else:
            if subtitle:
                subtitle += ' ' + line  # Concatenate with a space
            else:
                subtitle = line  # Initialize with the first line of text
    
    return subtitle  # Return the complete string of subtitles

def preprocess_multimodal(
    conversation_list: Sequence[str],
    multimodal_cfg: dict,
    cur_token_len: int,
    msg='',
    id=""
) -> Dict:
    # 将conversational list中
    is_multimodal = True
    # image_token_len = multimodal_cfg['image_token_len']
    image_token_len = cur_token_len
    subtitles = load_subtitles(id)
    split = conversation_list[0]["value"].split("<image>")
    conversation_list[0]["value"] = split[0] + "<Video>"+DEFAULT_IMAGE_PATCH_TOKEN * image_token_len +"</Video>" + "\n" + subtitles + split[1] + msg
    return [conversation_list]

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "###"
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = video_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = video_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation
        
def _tokenize_fn(strings: Sequence[str],
                tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{video_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_for_llama_v2(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    conv = copy.deepcopy(llama_v2_video_conversation.copy())
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    for source in sources:
        # <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n
        header = f"<s>[INST] <<SYS>>\n{conv.system}\n</SYS>>\n\n"

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        ).input_ids
    targets = copy.deepcopy(input_ids)


    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2 # 为什么减去2,speical token 的数目

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)
def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len
