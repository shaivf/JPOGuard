import argparse 
from pathlib import Path 
import subprocess 
import sys 

from dpo_runtime .data import read_json 
from dpo_runtime .reference_binding import sha 
from select_checkpoint import save_selection ,select 

CODE =Path (__file__ ).resolve ().parent 
GROUPS =("train","rewrite_error_only","original_only","mixed_90")


def sft_adapter (path ):
    path =path .resolve ()
    if path .is_dir ():
        return path 
    selected =read_json (path )["selected"]
    adapter =Path (selected ["adapter"]).resolve ()
    if sha (adapter /"adapter_model.safetensors")!=selected ["adapter_sha256"]:
        raise ValueError ("SFT adapter does not match its selection record")
    return adapter 


def train (data ,pairs ,base ,adapter ,output ,group ="train"):
    released =(pairs /"dpo_train.jsonl").is_file ()
    if released :
        from load_preferences import load_preferences 
        load_preferences (data ,pairs ,output /"pairs",adapter )
        pairs =output /"pairs"
    else :
        output .mkdir (parents =True ,exist_ok =False )
    dev_pairs =pairs /"dev.jsonl"
    train_pairs =pairs /(group +".jsonl")
    if not train_pairs .is_file ()or not dev_pairs .is_file ():
        raise ValueError (f"Missing generated DPO train/dev files for group {group }")
    dev_ref =output /"reference_dev.jsonl"
    train_ref =output /"reference_train.jsonl"
    policy =output /"policy"
    command =[sys .executable ,"-B","-m","dpo_runtime.train"]
    options =[
    "--src-fixed",str (CODE /"model"),"--base-model",str (base ),
    "--data",str (data ),
    ]
    if released :
        options .append ("--use-released-pairs")

    def run (mode ,chosen_adapter ,dataset ,destination ,*flags ):
        subprocess .run ([
        *command ,mode ,*options ,"--adapter",str (chosen_adapter ),"--pairs",str (dataset ),
        "--out",str (destination ),*flags ,
        ],cwd =CODE ,check =True )

    run ("reference",adapter ,train_pairs ,train_ref ,"--pair-split","train")
    run ("reference",adapter ,dev_pairs ,dev_ref ,"--pair-split","dev")
    dev_flags =("--pair-split","dev","--reference",str (dev_ref ),
    "--reference-adapter",str (adapter ),"--beta","0.1")
    check =output /"eval_sft.json"
    run ("evaluate",adapter ,dev_pairs ,check ,*dev_flags )
    if any (abs (row ["margin"])>=1e-3 for row in read_json (check )["records"]):
        raise RuntimeError ("Step-zero policy and reference disagree")
    run ("train",adapter ,train_pairs ,policy ,"--reference",str (train_ref ),
    "--epochs","1","--lr","5e-6","--beta","0.1","--seed","42",
    "--save-steps","50","--log-steps","10")
    candidates =sorted (policy .glob ("checkpoint-*"),key =lambda p :int (p .name .split ("-")[-1 ]))
    candidates .append (policy /"final")
    for candidate in candidates :
        name =candidate .name .replace ("-","")
        run ("evaluate",candidate ,dev_pairs ,output /("eval_"+name +".json"),*dev_flags )
    selection =select (output )
    destination =output /"selected_dpo_checkpoint.json"
    save_selection (selection ,destination )
    return destination 


def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--data",type =Path ,required =True )
    parser .add_argument ("--pairs-dir",type =Path ,required =True )
    parser .add_argument ("--base-model",type =Path ,required =True )
    parser .add_argument ("--sft-adapter",type =Path ,required =True )
    parser .add_argument ("--output",type =Path ,required =True )
    parser .add_argument ("--group",choices =GROUPS ,default ="train")
    args =parser .parse_args ()
    train (args .data .resolve (),args .pairs_dir .resolve (),args .base_model .resolve (),sft_adapter (args .sft_adapter ),
    args .output .resolve (),args .group )


if __name__ =="__main__":
    main ()
