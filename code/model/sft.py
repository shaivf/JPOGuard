import torch 
from torch .nn .utils .rnn import pad_sequence 
from transformers import AutoModelForImageTextToText ,AutoProcessor ,AutoTokenizer ,BitsAndBytesConfig 
from peft import LoraConfig ,TaskType ,get_peft_model ,prepare_model_for_kbit_training ,PeftModel 
from qwen_vl_utils import process_vision_info 

tokenizer =None 
processor =None 


def process_func (example ):
    ''
    MAX_LENGTH =8192 
    input_ids ,attention_mask ,labels =[],[],[]
    conversation =example ["conversations"]
    input_content =conversation [0 ]["value"]
    output_content =conversation [1 ]["value"]
    file_path =input_content .split ("<|vision_start|>")[1 ].split ("<|vision_end|>")[0 ]
    text_content =input_content .split ("<|vision_start|>")[0 ]
    messages =[
    {
    "role":"user",
    "content":[
    {
    "type":"image",
    "image":f"{file_path }",
    "resized_height":280 ,
    "resized_width":280 ,
    },
    {"type":"text","text":text_content },
    ],
    }
    ]
    text =processor .apply_chat_template (
    messages ,tokenize =False ,add_generation_prompt =True 
    )
    image_inputs ,video_inputs =process_vision_info (messages )
    inputs =processor (
    text =[text ],
    images =image_inputs ,
    videos =video_inputs ,
    padding =True ,
    return_tensors ="pt",
    )
    inputs ={key :value .tolist ()for key ,value in inputs .items ()}
    instruction =inputs 

    response =tokenizer (f"{output_content }",add_special_tokens =False )

    input_ids =(
    instruction ["input_ids"][0 ]+response ["input_ids"]+[tokenizer .pad_token_id ]
    )

    attention_mask =instruction ["attention_mask"][0 ]+response ["attention_mask"]+[1 ]
    labels =(
    [-100 ]*len (instruction ["input_ids"][0 ])
    +response ["input_ids"]
    +[tokenizer .pad_token_id ]
    )
    if not response ["input_ids"]:
        raise ValueError (f"SFT response has no supervised tokens: {file_path }")
    if len (input_ids )>MAX_LENGTH :
        raise ValueError (f"SFT sequence exceeds {MAX_LENGTH }: image={file_path }, "
        f"prompt={len (instruction ['input_ids'][0 ])}, "
        f"response={len (response ['input_ids'])}, total={len (input_ids )}; "
        "no silent truncation is allowed")

    input_ids =torch .tensor (input_ids )
    attention_mask =torch .tensor (attention_mask )
    labels =torch .tensor (labels )
    inputs ["pixel_values"]=torch .tensor (inputs ["pixel_values"])
    inputs ["image_grid_thw"]=torch .tensor (inputs ["image_grid_thw"]).squeeze (0 )
    return {
    "input_ids":input_ids ,
    "attention_mask":attention_mask ,
    "labels":labels ,
    "pixel_values":inputs ["pixel_values"],
    "image_grid_thw":inputs ["image_grid_thw"],
    }


class MultiModalCollator :
    def __init__ (self ,tokenizer ):
        self .tokenizer =tokenizer 

    @staticmethod 
    def _as_tensor (x ):
        ''
        return x if torch .is_tensor (x )else torch .tensor (x )

    def __call__ (self ,features ):
        t =self ._as_tensor 
        input_ids =[t (f ["input_ids"])for f in features ]
        attention_mask =[t (f ["attention_mask"])for f in features ]
        labels =[t (f ["labels"])for f in features ]
        pixel_values =[t (f ["pixel_values"])for f in features ]
        image_grid_thw =[t (f ["image_grid_thw"])for f in features ]

        input_ids =pad_sequence (
        input_ids ,
        batch_first =True ,
        padding_value =self .tokenizer .pad_token_id ,
        )
        attention_mask =pad_sequence (
        attention_mask ,
        batch_first =True ,
        padding_value =0 ,
        )
        labels =pad_sequence (
        labels ,
        batch_first =True ,
        padding_value =-100 ,
        )

        pixel_values =torch .cat (pixel_values ,dim =0 )
        image_grid_thw =torch .stack (image_grid_thw ,dim =0 )

        return {
        "input_ids":input_ids ,
        "attention_mask":attention_mask ,
        "labels":labels ,
        "pixel_values":pixel_values ,
        "image_grid_thw":image_grid_thw ,
        }


def build_model_processor_tokenizer (args ):
    global tokenizer ,processor 

    tokenizer =AutoTokenizer .from_pretrained (
    args .model_path ,
    use_fast =False ,
    trust_remote_code =True ,
    )

    processor =AutoProcessor .from_pretrained (args .model_path )

    if args .use_qlora :
        bnb_config =BitsAndBytesConfig (
        load_in_4bit =True ,
        bnb_4bit_quant_type ="nf4",
        bnb_4bit_compute_dtype =torch .bfloat16 ,
        bnb_4bit_use_double_quant =True ,
        )
        model =AutoModelForImageTextToText .from_pretrained (
        args .model_path ,
        quantization_config =bnb_config ,
        device_map ="auto",
        torch_dtype =torch .bfloat16 ,
        trust_remote_code =True ,
        )
        model =prepare_model_for_kbit_training (model ,use_gradient_checkpointing =True )
        print ("[qlora] 已启用 4-bit 量化基座")
    else :
        model =AutoModelForImageTextToText .from_pretrained (
        args .model_path ,
        device_map ="auto",
        torch_dtype =torch .bfloat16 ,
        trust_remote_code =True ,
        )
    model .enable_input_require_grads ()

    if args .resume_adapter :
    
        print (f"[resume] 从已有 adapter 继续训练: {args .resume_adapter }")
        model =PeftModel .from_pretrained (model ,args .resume_adapter ,is_trainable =True )
        return model 

    lora_config =LoraConfig (
    task_type =TaskType .CAUSAL_LM ,
    target_modules =[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    ],
    inference_mode =False ,
    r =args .lora_r ,
    lora_alpha =args .lora_alpha ,
    lora_dropout =args .lora_dropout ,
    bias ="none",
    )

    model =get_peft_model (model ,lora_config )
    model .print_trainable_parameters ()
    return model 
