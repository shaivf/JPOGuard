import argparse 
import json 
import os 
from functools import partial 
from pathlib import Path 

from annotation_api import (
assess ,completion ,image_url ,render ,template ,
)
from dpo_runtime .data import append_jsonl ,load_unique_by_id ,read_json ,validate_partition 
from dpo_runtime .length_policy import length_check ,render_joint ,slice_brace 
from dpo_runtime .prompt import JOINT_PROMPT 
from experiment_splits import annotation ,append_session ,training_splits 


METHODS =("sft","hard","preferences")
HEADLINE_KEYS =("Rewritten_Headline","Rewriten_Headline","Rewriten_Caption")


def rewritten_headline (result ):
    for key in HEADLINE_KEYS :
        value =result .get (key )
        if isinstance (value ,str )and value .strip ():
            return value .strip ()
    return ""


def generate_reference (generate ,judge ,image ,news ):
    original =news ["caption"]
    words =len (original .split ())
    values =dict (ORIGINAL_WORDS =words ,MAX_WORDS =words +3 )
    initial =render (template ("reference_headline"),**values ,NEWS_HEADLINE =original ,
    NEWS_CONTEXT =news ["article_context"],
    MISLEADING_REASON =annotation (news )["Reason"])
    prompt =initial 
    for index in range (9 ):
        title =rewritten_headline (json .loads (slice_brace (generate (image ,prompt ))))
        if not title :
            prompt =initial 
            continue 
        checked =length_check (original ,title )
        if not checked ["within_budget"]:
            failure ={"kind":"over_budget","reason":"New candidate exceeds the headline word budget.",
            "length":checked }
        elif title .casefold ()==original .strip ().casefold ():
            failure ={"kind":"unchanged_original","reason":"New candidate repeats the original misleading headline."}
        else :
            label ,reason =assess (judge ,image ,title ,news )
            if label =="No":
                return title 
            failure ={"kind":"still_misleading","reason":reason }
        feedback =json .dumps ({"new_candidate_to_revise":title ,"validation_failure":failure },
        ensure_ascii =False )
        prompt =initial +"\n\n"+render (template ("headline_revision"),**values ,
        VALIDATION_FEEDBACK_JSON =feedback )
        if index >=2 :
            prompt +="\n\n"+render (template ("headline_recovery"),**values ,
            RECOVERY_TARGET_WORDS =max (1 ,words +1 ))
        if index >=5 :
            prompt +="\n\n"+render (template ("headline_targeted_repair"),**values )
    return ""


def sft_target (generate ,judge ,image ,news ):
    clause =("- The generated news headline may contain at most 3 more words than the original headline.\n"
    "- Original headline word count: {{ORIGINAL_WORDS}}. Maximum rewritten headline word count: {{MAX_WORDS}}.")
    initial =JOINT_PROMPT .replace (clause ,"- There is no length limit; rewrite freely if needed.")
    if initial ==JOINT_PROMPT :
        raise ValueError ("Cannot locate the length instruction in the joint template")
    result =json .loads (slice_brace (generate (image ,render_joint (news ,initial ))))
    title =rewritten_headline (result )
    if (not title or not length_check (news ["caption"],title )["within_budget"]
    or title .casefold ()==news ["caption"].strip ().casefold ()
    or assess (judge ,image ,title ,news )[0 ]!="No"):
        title =generate_reference (generate ,judge ,image ,news )
    if not title :
        raise ValueError ("No valid SFT rewrite after bounded generation")
    reason =result .get ("Reason")
    return {"chosen":title ,"reason":reason .strip ()if isinstance (reason ,str )and reason .strip ()
    else annotation (news )["Reason"]}


def preference_votes (complete ,image ,news ,chosen ,candidates ):
    def judge (title ):
        return {"headline":title ,"votes":[assess (complete ,image ,title ,news )[0 ]for _ in range (3 )]}

    return {"chosen":judge (chosen ),"candidates":[
    {"index":row ["index"],**judge (row ["headline"])}for row in candidates 
    ]}


def generate_annotations (method ,data ,output ,generate ,judge ,resume =False ,chosen_path =None ,candidates_path =None ):
    news =read_json (data /"train_data.json")
    images =data /"train_img"
    train ,validation ,hard =training_splits (news )
    test =read_json (data /"test_data.json")
    correction =read_json (data /"test_correction.json")
    validate_partition (set (news ),news ,test ,correction )
    if method =="sft":
        ids =train +validation
    else :
        ids =list (hard )
    if method =="preferences":
        chosen =load_unique_by_id (chosen_path )
        sampled =load_unique_by_id (candidates_path )
        if set (chosen )!=set (sampled )or set (chosen )!=set (ids ):
            raise ValueError ("Candidate and reference files must cover the hard pool")
        ids =[sid for sid in ids if chosen [sid ]["chosen"]]

    finished =append_session (output ,resume )
    if not finished .keys ()<=set (ids ):
        raise ValueError ("Output contains IDs outside the current stage")
    for sid in ids :
        if sid in finished :
            continue 
        record =news [sid ]
        if method =="sft" and annotation (record )["Misleading"]=="No":
            append_jsonl (output ,{"id":sid ,"Misleading":"No","reason":annotation (record )["Reason"],"chosen":""})
            continue
        image =image_url (images /f"{sid }.jpg")
        if method =="sft":
            fields =sft_target (generate ,judge ,image ,record )
            fields ["Misleading"]="Yes"
        elif method =="hard":
            fields ={"chosen":generate_reference (generate ,judge ,image ,record ),
            "split":hard [sid ],"reason":annotation (record )["Reason"]}
        else :
            fields =preference_votes (judge ,image ,record ,chosen [sid ]["chosen"],
            sampled [sid ]["candidates"])
        append_jsonl (output ,{"id":sid ,**fields })


def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("stage",choices =METHODS )
    parser .add_argument ("--data",required =True ,type =Path )
    parser .add_argument ("--output",required =True ,type =Path )
    parser .add_argument ("--generation-model")
    parser .add_argument ("--generation-endpoint")
    parser .add_argument ("--generation-options",default ='{}')
    parser .add_argument ("--judge-model")
    parser .add_argument ("--judge-endpoint")
    parser .add_argument ("--judge-options",default ='{}')
    parser .add_argument ("--chosen",type =Path )
    parser .add_argument ("--candidates",type =Path )
    parser .add_argument ("--resume",action ="store_true")
    args =parser .parse_args ()
    generate =None
    judge =None
    if args .stage in ("sft","hard"):
        key =os .environ .get ("GENERATION_API_KEY")
        if not args .generation_model or not args .generation_endpoint or not key :
            parser .error ("Generation requires --generation-model, --generation-endpoint and GENERATION_API_KEY")
        options =json .loads (args .generation_options )
        if not isinstance (options ,dict ): parser .error ("--generation-options must be a JSON object")
        generate =partial (completion ,args .generation_endpoint ,args .generation_model ,key ,parameters ={"max_tokens":8192 ,**options})
    if args .stage in ("sft","hard","preferences"):
        key =os .environ .get ("JUDGE_API_KEY")
        if not args .judge_model or not args .judge_endpoint or not key :
            parser .error ("Judging requires --judge-model, --judge-endpoint and JUDGE_API_KEY")
        options =json .loads (args .judge_options )
        if not isinstance (options ,dict ): parser .error ("--judge-options must be a JSON object")
        judge =partial (completion ,args .judge_endpoint ,args .judge_model ,key ,parameters ={"max_tokens":8192 ,**options})
    if args .stage =="preferences"and not (args .chosen and args .candidates ):
        parser .error ("preferences requires --chosen and --candidates")
    generate_annotations (args .stage ,args .data ,args .output ,generate ,judge ,args .resume ,
    args .chosen ,args .candidates )


if __name__ =="__main__":
    main ()
