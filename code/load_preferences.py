import hashlib 
from pathlib import Path 

from build_preferences import mixed_subset 
from dpo_runtime .data import load_unique_by_id ,pair_digest ,read_json ,write_json ,write_jsonl 
from dpo_runtime .length_policy import validate_pair_lengths 
from dpo_runtime .prompt import JOINT_PROMPT 
from dpo_runtime .reference_binding import sha 
from experiment_splits import annotation ,training_splits 


def load_preferences (data ,targets ,output ,adapter ):
    news =read_json (data /"train_data.json")
    _ ,_ ,hard =training_splits (news )
    groups ={}
    for split in ("train","dev"):
        entries =load_unique_by_id (targets /f"dpo_{split }.jsonl")
        rows =[]
        for sid ,entry in entries .items ():
            if hard .get (sid )!=split :
                raise ValueError (f"Incorrect preference split: {sid }")
            record =news [sid ]
            reason =entry .get ("reason",annotation (record )["Reason"])
            chosen ={"Misleading":"Yes","Reason":reason ,"Rewritten_Headline":entry ["chosen"]}
            rejected ={"Misleading":"Yes","Reason":reason ,
            "Rewritten_Headline":entry ["rejected"]}
            pair ={"id":sid ,"split":split ,
            "pair_type":"original_fallback"if entry ["rejected"].strip ()==record ["caption"].strip () else "rewrite_error",
            "chosen_response":chosen ,"rejected_response":rejected }
            validate_pair_lengths (record ,pair )
            pair ["pair_digest"]=pair_digest (pair )
            rows .append (pair )
        groups [split ]=rows 
    groups ["rewrite_error_only"]=[r for r in groups ["train"]if r ["pair_type"]=="rewrite_error"]
    groups ["original_only"]=[r for r in groups ["train"]if r ["pair_type"]=="original_fallback"]
    groups ["mixed_90"]=mixed_subset (groups ["train"],42 )
    output .mkdir (parents =True ,exist_ok =False )
    for name ,rows in groups .items ():
        path =output /f"{name }.jsonl"
        write_jsonl (path ,rows )
        write_json (path .with_suffix (".jsonl.length_policy.json"),{
        "extra_words":3 ,"fresh_candidates_verified":True ,
        "joint_template_sha256":hashlib .sha256 (JOINT_PROMPT .encode ()).hexdigest (),
        "pairs_sha256":sha (path ),"pair_split":"dev"if name =="dev"else "train",
        "pair_count":len (rows ),
        })
