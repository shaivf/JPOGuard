import argparse 
import hashlib 
from pathlib import Path 

from dpo_runtime .data import append_jsonl ,read_json ,validate_partition ,write_json 
from dpo_runtime .length_policy import length_check 
from dpo_runtime .prompt import JOINT_PROMPT ,joint_prompt 
from experiment_splits import append_session ,training_splits 
from model .runtime_io import parse_correction_response 
from train_dpo import sha ,sft_adapter 
from vl_decode import generate 


def sample_seed (seed ,sid ,index ):
    digest =hashlib .sha256 (f"{seed }:{sid }:{index }".encode ()).digest ()
    return int .from_bytes (digest [:4 ],"big")


def eligible_candidates (sample ,news ):
    candidates =[]
    for index in range (3 ):
        parsed =sample (index )
        if parsed ["status"]!="ok":
            continue 
        headline =parsed ["rewritten_headline"]
        check =length_check (news ["caption"],headline )
        if parsed ["pred_label"]=="Yes"and check ["nonempty"]and check ["within_budget"]:
            candidates .append ({"index":index ,"pred_label":"Yes","headline":headline })
    return candidates 


def sample_pool (data ,base ,adapter ,output ,resume ):
    import torch 
    from model .common import extract_json_obj ,load_model ,resolve_key 

    train_data =data /"train_data.json"
    news =read_json (train_data )
    _ ,_ ,hard =training_splits (news )
    validate_partition (set (hard ),news ,read_json (data /"test_data.json"),
    read_json (data /"test_correction.json"))
    identity ={
    "candidate_adapter_sha256":sha (adapter /"adapter_model.safetensors"),
    "train_data_sha256":sha (train_data ),
    "joint_template_sha256":hashlib .sha256 (JOINT_PROMPT .encode ()).hexdigest (),
    }
    provenance =output .with_suffix (output .suffix +".meta.json")
    finished =append_session (output ,resume )
    if provenance .exists ():
        if not resume or read_json (provenance )!=identity :
            raise ValueError ("Candidate sampling inputs changed")
    else :
        if finished :
            raise ValueError ("Missing existing candidate metadata")
        write_json (provenance ,identity )
    if not finished .keys ()<=hard .keys ():
        raise ValueError ("Candidate file contains IDs outside the hard pool")
    if finished .keys ()==hard .keys ():
        return 
    model ,processor ,_ =load_model (str (base ),str (adapter ))
    for sid ,split in hard .items ():
        if sid in finished :
            continue 
        record =news [sid ]
        image =data /"train_img"/f"{sid }.jpg"
        prompt =joint_prompt (record )
        if not image .is_file ():
            raise FileNotFoundError (image )

        def sample (index ):
            torch .manual_seed (sample_seed (42 ,sid ,index ))
            raw =generate (model ,processor ,prompt ,image ,1536 ,temperature =0.8 )["raw"]
            return parse_correction_response (raw ,extract_json_obj ,resolve_key )

        append_jsonl (output ,{"id":sid ,"split":split ,
        "candidates":eligible_candidates (sample ,record )})


def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--data",type =Path ,required =True )
    parser .add_argument ("--base-model",type =Path ,required =True )
    parser .add_argument ("--sft-adapter",type =Path ,required =True )
    parser .add_argument ("--output",type =Path ,required =True )
    parser .add_argument ("--resume",action ="store_true")
    args =parser .parse_args ()
    sample_pool (args .data .resolve (),args .base_model .resolve (),sft_adapter (args .sft_adapter ),
    args .output .resolve (),args .resume )


if __name__ =="__main__":
    main ()
