import json 
from pathlib import Path 
import re 

import torch 
from peft import PeftModel 
from transformers import AutoModelForImageTextToText ,AutoProcessor ,AutoTokenizer 

RESIZED =280 
_DASHES =("–","—","‒","‐","‑","-")

def normalize_key (s :str )->str :
    ''
    for d in _DASHES :
        s =s .replace (d ,"-")
    return s .strip ().lower ()


def resolve_key (d ,*candidates ,default =None ):
    ''
    if not isinstance (d ,dict ):
        return default 
    norm ={normalize_key (k ):v for k ,v in d .items ()}
    for c in candidates :
        if c in d :
            return d [c ]
        nk =normalize_key (c )
        if nk in norm :
            return norm [nk ]
    return default 


def strip_control_chars (s :str )->str :
    ''
    return re .sub (r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]","",s )


def extract_json_obj (s ):
    ''
    if not isinstance (s ,str ):
        return None 
    s =strip_control_chars (s )
    i ,j =s .find ("{"),s .rfind ("}")
    if i ==-1 or j ==-1 or j <=i :
        return None 
    frag =s [i :j +1 ]
    for candidate in (frag ,re .sub (r",\s*([}\]])",r"\1",frag )):
        try :
            obj =json .loads (candidate )
            if isinstance (obj ,dict ):
                return obj 
        except json .JSONDecodeError :
            continue 
    return None 


def _verify_lora_loaded (model )->tuple :
    ''
    nz =tot =0 
    for name ,p in model .named_parameters ():
        if "lora_B"in name :
            tot +=1 
            if p .abs ().sum ().item ()>0 :
                nz +=1 
    return nz ,tot 


def load_model (
base_model_path :str ,
adapter_path :str =None ,
device :str ="cuda:0",
is_trainable :bool =False ,
):
    print (f"[load] base={base_model_path }")
    tokenizer =AutoTokenizer .from_pretrained (base_model_path ,use_fast =False ,trust_remote_code =True )
    processor =AutoProcessor .from_pretrained (base_model_path )

    model =AutoModelForImageTextToText .from_pretrained (
    base_model_path ,
    device_map =device ,
    torch_dtype =torch .bfloat16 ,
    trust_remote_code =True ,
    )

    if adapter_path :
        print (f"[load] adapter={adapter_path }")
        if not (Path (adapter_path )/'adapter_config.json').is_file ():
            raise FileNotFoundError (Path (adapter_path )/'adapter_config.json')
        model =PeftModel .from_pretrained (model ,adapter_path ,is_trainable =is_trainable )

        nz ,tot =_verify_lora_loaded (model )
        print (f"[load] LoRA 校验: {nz }/{tot } 个 lora_B 非零")
        if tot ==0 :
            raise RuntimeError ("未找到任何 lora_B 参数 —— adapter 未生效,请检查 target_modules")
        if nz ==0 :
            raise RuntimeError (
            "所有 lora_B 均为零 —— 权重未成功加载(模型等价于基座)。"
            "请检查 checkpoint 目录与 base model 是否匹配。"
            )
        if nz <tot :
            print (f"[load] ⚠️ 有 {tot -nz } 个 lora_B 为零,可能部分层未加载")

    model .eval ()if not is_trainable else model .train ()
    return model ,processor ,tokenizer 
