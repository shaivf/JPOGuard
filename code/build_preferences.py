''

from __future__ import annotations 

import argparse 
import hashlib 
import json 
import random 
from collections import Counter 
from pathlib import Path 

from dpo_runtime .data import load_unique_by_id ,pair_digest ,read_json ,validate_partition ,write_json ,write_jsonl 
from dpo_runtime .length_policy import validate_pair_lengths 
from dpo_runtime .prompt import JOINT_PROMPT 
from dpo_runtime .reference_binding import sha 
from experiment_splits import annotation ,training_splits 


def votes_for (row :dict ,label :str )->int :
    return sum (vote ==label for vote in row .get ("votes",[]))


def gold_is_misleading (record :dict )->bool :
    return annotation (record )["Misleading"]=="Yes"


def fallback_rank (row :dict ,seed :int )->str :
    payload =f"{seed }:{row ['split']}:{row ['id']}"
    return hashlib .sha256 (payload .encode ("utf-8")).hexdigest ()


def mixed_subset (rows ,seed ):
    errors =sorted (row ["id"]for row in rows if row ["pair_type"]=="rewrite_error")
    originals =sorted (row ["id"]for row in rows if row ["pair_type"]=="original_fallback")
    if len (errors )<90 or len (originals )<45 :
        return []
    error90 =sorted (random .Random (seed ).sample (errors ,90 ))
    selected =set (random .Random (seed +1 ).sample (error90 ,45 ))
    selected .update (random .Random (seed +2 ).sample (originals ,45 ))
    return [row for row in rows if row ["id"]in selected ]


def build (data :Path ,output :Path ,chosen_path :Path ,candidates_path :Path ,
judged_path :Path ,required_votes :int =2 ,
max_fallback_ratio :int =3 ,seed :int =42 ):
    train_path =data /"train_data.json"
    train =read_json (train_path )
    test =read_json (data /"test_data.json")
    correction =read_json (data /"test_correction.json")
    chosen =load_unique_by_id (chosen_path )
    sampled =load_unique_by_id (candidates_path )
    judged =load_unique_by_id (judged_path )
    validate_partition (set (chosen )|set (sampled )|set (judged ),train ,test ,correction )
    _ ,_ ,hard =training_splits (train )
    if (chosen .keys ()!=hard .keys ()or sampled .keys ()!=hard .keys ()
    or judged .keys ()!={sid for sid ,row in chosen .items ()if row ["chosen"]}
    or any (hard [sid ]!=chosen [sid ]["split"]or hard [sid ]!=sampled [sid ]["split"]for sid in hard )):
        raise ValueError ("Input IDs and splits must match the hard pool and all available titles must be judged")
    metadata =read_json (candidates_path .with_suffix (candidates_path .suffix +".meta.json"))
    prompt_sha =hashlib .sha256 (JOINT_PROMPT .encode ()).hexdigest ()
    adapter_sha =metadata ["candidate_adapter_sha256"]
    if (metadata ["train_data_sha256"]!=sha (train_path )
    or metadata ["joint_template_sha256"]!=prompt_sha or not adapter_sha ):
        raise ValueError ("Candidate metadata does not match the training data and joint prompt")
    if required_votes not in (2 ,3 ):
        raise ValueError ("required_votes must be 2 or 3 out of 3")
    if max_fallback_ratio <0 :
        raise ValueError ("max_fallback_ratio must be non-negative")

    available ={
    "train":{"rewrite_error":[],"original_fallback":[]},
    "dev":{"rewrite_error":[],"original_fallback":[]},
    }
    skipped =Counter ()
    available_types =Counter ()
    for sample_id ,judged_row in judged .items ():
        if judged_row .get ("error"):
            skipped ["judge_error"]+=1 
            continue 
        if not gold_is_misleading (train [sample_id ]):
            raise ValueError (f"original is not gold misleading: {sample_id }")
        teacher =judged_row .get ("chosen")or {}
        teacher_headline =chosen [sample_id ]["chosen"].strip ()
        original_headline =str (train [sample_id ].get ("caption")or "").strip ()
        if teacher .get ("headline")!=teacher_headline :
            raise ValueError (f"chosen headline mismatch: {sample_id }")
        if len (teacher .get ("votes",[]))!=3 or votes_for (teacher ,"No")<required_votes :
            skipped ["chosen_not_confirmed"]+=1 
            continue 
        if not original_headline or original_headline .casefold ()==teacher_headline .casefold ():
            skipped ["invalid_original_fallback"]+=1 
            continue 
        reason =str (chosen [sample_id ].get ("reason")or "").strip ()
        if not reason :
            skipped ["empty_chosen_reason"]+=1 
            continue 

        chosen_response ={
        "Misleading":"Yes",
        "Reason":reason ,
        "Rewritten_Headline":teacher_headline ,
        }
        original_candidates =sampled [sample_id ].get ("candidates")or []
        judged_candidates ={
        str (row .get ("index")):row for row in judged_row .get ("candidates")or []
        }
        if len (judged_candidates )!=len (judged_row .get ("candidates")or []):
            raise ValueError (f"duplicate candidate vote indices: {sample_id }")
        if set (judged_candidates )!={str (row .get ("index"))for row in original_candidates }:
            raise ValueError (f"candidate vote coverage mismatch: {sample_id }")
        for candidate in original_candidates :
            vote_row =judged_candidates [str (candidate .get ("index"))]
            if str (vote_row .get ("headline")or "").strip ()!=str (candidate .get ("headline")or "").strip ():
                raise ValueError (f"candidate/vote headline mismatch: {sample_id }")

        selected_type =None 
        rejected_response =None 
        confirmed_good_rewrite =False 



        for candidate in sorted (original_candidates ,key =lambda row :row ["index"]):
            if str (candidate .get ("pred_label")or "").strip ().lower ()!="yes":
                continue 
            headline =str (candidate .get ("headline")or "").strip ()
            if not headline or headline .casefold ()==teacher_headline .casefold ():
                continue 
            vote_row =judged_candidates [str (candidate ["index"])]
            votes =vote_row .get ("votes")or []
            if len (votes )!=3 :
                continue 
            if votes_for (vote_row ,"Yes")>=required_votes and selected_type is None :
                selected_type ="rewrite_error"
                rejected_response ={
                "Misleading":"Yes",
                "Reason":reason ,
                "Rewritten_Headline":headline ,
                }
            if votes_for (vote_row ,"No")>=required_votes :
                confirmed_good_rewrite =True 

        if selected_type is None :
            if not confirmed_good_rewrite :
                skipped ["no_confirmed_correction_candidate"]+=1 
                continue 
            selected_type ="original_fallback"
            rejected_response ={
            "Misleading":"Yes",
            "Reason":reason ,
            "Rewritten_Headline":original_headline ,
            }

        if chosen_response ==rejected_response :
            raise ValueError (f"identical chosen and rejected responses: {sample_id }")
        split =chosen [sample_id ]["split"]
        if split not in available :
            raise ValueError (f"unexpected split: {sample_id } {split }")
        row ={
        "id":sample_id ,
        "split":split ,
        "pair_type":selected_type ,
        "chosen_response":chosen_response ,
        "rejected_response":rejected_response ,
        }
        validate_pair_lengths (train [sample_id ],row )
        row ["pair_digest"]=pair_digest (row )
        available [split ][selected_type ].append (row )
        available_types [selected_type ]+=1 

    pairs ={"train":[],"dev":[]}
    for split in ("train","dev"):
        genuine =available [split ]["rewrite_error"]
        fallback =sorted (
        available [split ]["original_fallback"],
        key =lambda row :fallback_rank (row ,seed ),
        )
        fallback_limit =len (genuine )*max_fallback_ratio 
        kept_fallback =fallback [:fallback_limit ]
        skipped ["fallback_ratio_cap"]+=len (fallback )-len (kept_fallback )
        pairs [split ]=sorted (genuine +kept_fallback ,key =lambda row :row ["id"])

    types =Counter (row ["pair_type"]for rows in pairs .values ()for row in rows )
    if not pairs ["train"]or not any (row ["pair_type"]=="rewrite_error"for row in pairs ["dev"]):
        raise ValueError ("DPO requires training pairs and a development rewrite-error subset")
    groups ={
    **pairs ,
    "rewrite_error_only":[row for row in pairs ["train"]if row ["pair_type"]=="rewrite_error"],
    "original_only":[row for row in pairs ["train"]if row ["pair_type"]=="original_fallback"],
    "mixed_90":mixed_subset (pairs ["train"],seed ),
    }
    output .mkdir (parents =True ,exist_ok =False )
    for name ,rows in groups .items ():
        if not rows :
            continue 
        path =output /f"{name }.jsonl"
        write_jsonl (path ,rows )
        write_json (path .with_suffix (".jsonl.length_policy.json"),{
        "candidate_adapter_sha256":adapter_sha ,
        "extra_words":3 ,
        "fresh_candidates_verified":True ,
        "joint_template_sha256":prompt_sha ,
        "new_sft_adapter_sha256":adapter_sha ,
        "pair_count":len (rows ),
        "pair_split":"dev"if name =="dev"else "train",
        "pairs_sha256":sha (path ),
        })
    summary ={
    "processed":len (judged ),
    "train_pairs":len (pairs ["train"]),
    "dev_pairs":len (pairs ["dev"]),
    "by_type":dict (types ),
    "available_by_type":dict (available_types ),
    "skipped":dict (skipped ),
    "required_votes":required_votes ,
    "max_fallback_ratio":max_fallback_ratio ,
    "seed":seed ,
    "correction_only":True ,
    "test_overlap":0 ,
    "mixed_90_pairs":len (groups ["mixed_90"]),
    }
    (output /"pair_summary.json").write_text (
    json .dumps (summary ,ensure_ascii =False ,indent =2 )+"\n",encoding ="utf-8"
    )
    return summary 


def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--data",type =Path ,required =True )
    parser .add_argument ("--out",type =Path ,required =True )
    parser .add_argument ("--chosen",type =Path ,required =True )
    parser .add_argument ("--candidates",type =Path ,required =True )
    parser .add_argument ("--judged",type =Path ,required =True )
    parser .add_argument ("--required-votes",type =int ,choices =(2 ,3 ),default =2 )
    parser .add_argument ("--max-fallback-ratio",type =int ,default =3 )
    parser .add_argument ("--seed",type =int ,default =42 )
    args =parser .parse_args ()
    if args .out .exists ():
        parser .error ("--out must be a new directory")
    print (json .dumps (build (
    args .data .resolve (),args .out .resolve (),args .chosen ,args .candidates ,args .judged ,
    args .required_votes ,args .max_fallback_ratio ,args .seed ,
    ),ensure_ascii =False ))


if __name__ =="__main__":
    main ()
