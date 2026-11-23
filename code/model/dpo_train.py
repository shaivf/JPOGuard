import torch 
import torch .nn .functional as F 
from common import RESIZED 


def encode (model ,processor ,tokenizer ,prompt ,image_path ,response ):
    ''
    from qwen_vl_utils import process_vision_info 

    messages =[{
    "role":"user",
    "content":[
    {"type":"image","image":image_path ,
    "resized_height":RESIZED ,"resized_width":RESIZED },
    {"type":"text","text":prompt },
    ],
    }]
    
    
    
    
    
    text =processor .apply_chat_template (
    messages ,tokenize =False ,add_generation_prompt =True )
    image_inputs ,video_inputs =process_vision_info (messages )
    inputs =processor (
    text =[text ],images =image_inputs ,videos =video_inputs ,
    padding =True ,return_tensors ="pt")

    resp =tokenizer (response ,add_special_tokens =False ,return_tensors ="pt")
    input_ids =torch .cat ([inputs ["input_ids"],resp ["input_ids"]],dim =1 )
    attention_mask =torch .cat ([inputs ["attention_mask"],resp ["attention_mask"]],dim =1 )
    labels =torch .cat ([
    torch .full_like (inputs ["input_ids"],-100 ),
    resp ["input_ids"],
    ],dim =1 )
    if input_ids .shape [1 ]>8192 :
        raise ValueError (
        f"encoded sequence exceeds SFT MAX_LENGTH=8192: {input_ids .shape [1 ]}"
        )
    return input_ids ,attention_mask ,labels ,inputs ["pixel_values"],inputs ["image_grid_thw"]


def compute_logps (model ,input_ids ,attention_mask ,labels ,pixel_values ,image_grid_thw ,device ):
    ''
    out =model (input_ids =input_ids .to (device ),
    attention_mask =attention_mask .to (device ),
    pixel_values =pixel_values .to (device ),
    image_grid_thw =image_grid_thw .to (device ))
    logits =out .logits [:,:-1 ,:]
    tgt =labels [:,1 :].to (device )
    mask =(tgt !=-100 ).float ()
    logp =F .log_softmax (logits .float (),dim =-1 )
    gathered =logp .gather (2 ,tgt .clamp (min =0 ).unsqueeze (-1 )).squeeze (-1 )
    return (gathered *mask ).sum ()
