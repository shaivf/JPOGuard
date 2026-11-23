import argparse 
from pathlib import Path 
import json 
import sys 

from development_metrics import label_prefix 
from dpo_runtime .data import append_jsonl ,load_unique_by_id ,read_json ,write_json 
from dpo_runtime .length_policy import length_check 
from dpo_runtime .prompt import joint_prompt 
from model .runtime_io import parse_correction_response 
from vl_decode import generate 

CODE =Path (__file__ ).resolve ().parent 
VIEWS =("Reader_View_ih","Reader_View_context")


def test_inputs (data ,reader_data ):
    original =read_json (data /"test_data.json")
    views =load_unique_by_id (reader_data )
    correction =read_json (data /"test_correction.json")
    if len (original )!=1000 or len (correction )!=263 or not correction .keys ()<=views .keys ()==original .keys ():
        raise ValueError ("Unexpected official test split")
    actual ={}
    for sid ,record in original .items ():
        values ={}
        for key in VIEWS:
            value =views [sid ].get (key )
            if isinstance (value ,dict ) and set (value )=={"article_context_parts"}:
                value =record ["article_context"].strip ().join (value ["article_context_parts"])
            values [key ]=value
        if not all (isinstance (values [key ],str )and values [key ].strip ()for key in VIEWS ):
            raise ValueError (f"Missing reader input for {sid }")
        if "caption" in views [sid ] and (views [sid ]["caption"]!=record ["caption"]
        or views [sid ]["Misleading"]!=json .loads (record ["Misleading"])["Misleading"]
        or views [sid ]["correction"]!=(sid in correction )):
            raise ValueError (f"Test annotation mismatch for {sid }")
        if not (data /"test_img"/(sid +".jpg")).is_file ():
            raise FileNotFoundError (sid )
        actual [sid ]={**record ,**values }
    if any (json .loads (original [sid ]["Misleading"])["Misleading"]!="Yes"for sid in correction ):
        raise ValueError ("Correction split contains a non-misleading item")
    return actual ,correction ,data /"test_img"


def evaluate (test_set ,base ,adapter ,output ):
    sys .path .insert (0 ,str (CODE /"model"))
    from common import load_model ,extract_json_obj ,resolve_key 

    news ,correction ,images =test_set 
    output .mkdir (parents =True ,exist_ok =False )
    model ,processor ,_ =load_model (str (base ),str (adapter ))
    correct =0 
    for sid ,record in news .items ():
        generation =generate (model ,processor ,joint_prompt (record ),images /(sid +".jpg"),64 ,True )
        predicted =label_prefix (generation ["raw"])
        gold =json .loads (record ["Misleading"])["Misleading"]
        correct +=predicted ==gold 
        append_jsonl (output /"detection.jsonl",{"id":sid ,"pred":predicted ,"gold":gold ,**generation })
    for sid in correction :
        record =news [sid ]
        generation =generate (model ,processor ,joint_prompt (record ),images /(sid +".jpg"),1536 )
        prediction =parse_correction_response (generation ["raw"],extract_json_obj ,resolve_key )
        if generation ["hit_token_budget"]and not generation ["eos_observed"]:
            prediction ["status"]="truncated"
        if prediction ["status"]=="ok":
            prediction ["length"]=length_check (record ["caption"],prediction ["rewritten_headline"])
        append_jsonl (output /"correction.jsonl",{"id":sid ,**generation ,**prediction })
        if prediction ["status"]!="ok":
            raise ValueError (f"Incomplete correction response: {sid }")
    write_json (output /"detection_summary.json",
    {"correct":correct ,"total":len (news ),"accuracy":correct /len (news )})


def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--data",type =Path ,required =True )
    parser .add_argument ("--reader-data",type =Path ,required =True )
    parser .add_argument ("--base-model",type =Path )
    parser .add_argument ("--adapter",type =Path )
    parser .add_argument ("--output",type =Path )
    parser .add_argument ("--check-data",action ="store_true")
    args =parser .parse_args ()
    test_set =test_inputs (args .data .resolve (),args .reader_data .resolve ())
    if args .check_data :
        news ,correction ,_ =test_set 
        print (len (news ),len (correction ))
        return 
    if not all ((args .base_model ,args .adapter ,args .output )):
        parser .error ("--base-model, --adapter and --output are required")
    from train_dpo import sft_adapter 
    evaluate (test_set ,args .base_model .resolve (),sft_adapter (args .adapter ),
    args .output .resolve ())


if __name__ =="__main__":
    main ()
