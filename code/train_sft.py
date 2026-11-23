import argparse 
import json 
from pathlib import Path 
from types import SimpleNamespace 
import math 

from dpo_runtime .data import load_unique_by_id ,read_json 
from dpo_runtime .length_policy import length_check 
from dpo_runtime .prompt import joint_prompt 
from dpo_runtime .reference_binding import sha 
from experiment_splits import annotation ,training_splits 


def load_splits (data ,targets_path ):
    news =read_json (data /"train_data.json")
    train_ids ,dev_ids ,_ =training_splits (news )
    targets =load_unique_by_id (targets_path )
    expected =set (train_ids +dev_ids)
    if targets .keys ()!=expected :
        raise ValueError (f"Expected {len (expected )} SFT targets")
    by_id ={}
    for sid in train_ids +dev_ids :
        record =news [sid ]
        gold =annotation (record )
        target =targets [sid]
        reason =target ["reason"]
        headline =target ["chosen"]
        check =length_check (record ["caption"],headline )
        if (target ["Misleading"]!=gold ["Misleading"] or not reason .strip ()
        or (gold ["Misleading"]=="Yes" and (not check ["nonempty"] or not check ["within_budget"]))
        or (gold ["Misleading"]=="No" and headline)):
            raise ValueError (f"Invalid SFT supervision: {sid }")
        image =(data /"train_img"/f"{sid }.jpg").resolve ()
        if not image .is_file ():
            raise FileNotFoundError (image )
        response ={"Misleading":gold ["Misleading"],"Reason":reason ,"Rewritten_Headline":headline }
        by_id [sid ]={"id":sid ,"task":"Misleading_Detection_And_Correction","conversations":[
        {"from":"user","value":f"{joint_prompt (record ).rstrip ()}\n\n: <|vision_start|>{image }<|vision_end|>"},
        {"from":"assistant","value":json .dumps (response ,ensure_ascii =False ,indent =4 )},
        ]}
    return [by_id [sid ]for sid in train_ids ],[by_id [sid ]for sid in dev_ids ]


def train (data ,targets ,base_model ,initial_adapter ,output ):
    import torch 
    from datasets import Dataset 
    from transformers import Trainer ,TrainerCallback ,TrainingArguments 
    from model import sft 
    from dpo_runtime .data import write_json 

    dataset_train ,dataset_dev =load_splits (data ,targets )
    for file in (base_model /"config.json",initial_adapter /"adapter_config.json",
    initial_adapter /"adapter_model.safetensors"):
        if not file .is_file ():
            raise FileNotFoundError (file )
    output .mkdir (parents =True ,exist_ok =False )
    torch .manual_seed (42 )
    model =sft .build_model_processor_tokenizer (SimpleNamespace (
    model_path =str (base_model ),resume_adapter =str (initial_adapter ),use_qlora =True ))
    model .config .use_cache =False 
    if any ("lora_"not in name for name ,value in model .named_parameters ()if value .requires_grad ):
        raise RuntimeError ("Expected only trainable LoRA parameters")
    train_set ,dev_set =(Dataset .from_list (rows ).map (
    sft .process_func ,remove_columns =list (rows [0 ]))for rows in (dataset_train ,dataset_dev ))
    scores =[]

    class KeepAdapters (TrainerCallback ):
        def on_save (self ,args ,state ,control ,**kwargs ):
            step =state .global_step 
            losses =[row ["eval_loss"]for row in state .log_history 
            if row .get ("step")==step and "eval_loss"in row ]
            if losses :
                candidate =output /"candidates"/f"checkpoint-{step }"
                kwargs ["model"].save_pretrained (str (candidate ))
                scores .append ((losses [-1 ],step ,candidate ))
            return control 

    args =TrainingArguments (
    output_dir =str (output ),per_device_train_batch_size =1 ,per_device_eval_batch_size =2 ,
    gradient_accumulation_steps =4 ,num_train_epochs =2 ,learning_rate =1e-4 ,
    weight_decay =0.0 ,warmup_steps =0 ,warmup_ratio =0.0 ,logging_steps =10 ,
    save_steps =200 ,eval_steps =200 ,save_total_limit =2 ,seed =42 ,data_seed =42 ,
    eval_strategy ="steps",save_strategy ="steps",load_best_model_at_end =False ,
    gradient_checkpointing =True ,report_to ="none",remove_unused_columns =False ,bf16 =True ,
    )
    trainer =Trainer (model =model ,args =args ,train_dataset =train_set ,eval_dataset =dev_set ,
    data_collator =sft .MultiModalCollator (sft .tokenizer ),callbacks =[KeepAdapters ()])
    trainer .train ()
    final =output /"final"
    trainer .save_model (str (final ))
    sft .processor .save_pretrained (str (final ))
    sft .tokenizer .save_pretrained (str (final ))
    scores .append ((trainer .evaluate ()["eval_loss"],trainer .state .global_step ,final ))
    if any (not math .isfinite (loss )for loss ,_ ,_ in scores ):
        raise RuntimeError ("Non-finite SFT validation loss")
    candidates =[{
    "name":adapter .name ,"adapter":str (adapter ),"eval_loss":loss ,"step":step ,
    "adapter_sha256":sha (adapter /"adapter_model.safetensors"),
    }for loss ,step ,adapter in sorted (scores )]
    selection ={"selected":candidates [0 ],"candidates":candidates }
    write_json (output /"selected_sft_checkpoint.json",selection )
    return selection 


def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--data",type =Path ,required =True )
    parser .add_argument ("--targets",type =Path ,required =True )
    parser .add_argument ("--base-model",type =Path )
    parser .add_argument ("--initial-adapter",type =Path )
    parser .add_argument ("--output",type =Path )
    parser .add_argument ("--check-data",action ="store_true")
    args =parser .parse_args ()
    if args .check_data :
        train_rows ,dev_rows =load_splits (args .data .resolve (),args .targets .resolve ())
        print (len (train_rows ),len (dev_rows ))
        return 
    if not all ((args .base_model ,args .initial_adapter ,args .output )):
        parser .error ("--base-model, --initial-adapter and --output are required")
    train (args .data .resolve (),args .targets .resolve (),args .base_model .resolve (),args .initial_adapter .resolve (),
    args .output .resolve ())


if __name__ =="__main__":
    main ()
