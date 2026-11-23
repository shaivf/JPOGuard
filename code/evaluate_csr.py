import argparse 
import json
import os 
from pathlib import Path 

from annotation_api import assess ,completion ,image_url 
from dpo_runtime .data import load_unique_by_id ,read_json ,write_json 
from dpo_runtime .length_policy import length_check 


def predict_verdict (complete ,image ,title ,record ):
    encoded =image_url (image )
    return [assess (complete ,encoded ,title ,record )[0 ]for _ in range (3 )]


def evaluate (data ,predictions ,output ,complete ):
    original =read_json (data /"test_data.json")
    correction =read_json (data /"test_correction.json")
    rows =load_unique_by_id (predictions )
    if len (correction )!=263 or set (rows )!=set (correction ):
        raise ValueError ("Incomplete official correction predictions")
    if any (row ["status"]!="ok"for row in rows .values ()):
        raise ValueError ("Invalid correction predictions")
    output .mkdir (parents =True ,exist_ok =False )
    results ={}
    for sid ,row in rows .items ():
        title =row ["rewritten_headline"]
        record =original [sid ]
        length =length_check (record ["caption"],title )
        judgments =[]
        if not length ["nonempty"]:
            failure ="empty_rewrite"
        elif not length ["within_budget"]:
            failure ="overlength"
        else :
            judgments =predict_verdict (complete ,data /"test_img"/(sid +".jpg"),title ,record )
            failure =None if judgments .count ("No")>=2 else "still_misleading"
        results [sid ]={"id":sid ,"rewritten_headline":title ,"csr_success":failure is None ,
        "failure_kind":failure ,"votes":judgments }
    correct =sum (row ["csr_success"]for row in results .values ())
    write_json (output /"judgments.json",results )
    write_json (output /"csr.json",{"success":correct ,"total":len (results ),"csr":correct /len (results )})


def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--data",type =Path ,required =True )
    parser .add_argument ("--predictions",type =Path ,required =True )
    parser .add_argument ("--output",type =Path ,required =True )
    parser .add_argument ("--judge-model",required =True )
    parser .add_argument ("--judge-endpoint",required =True )
    parser .add_argument ("--judge-options",default ='{}')
    args =parser .parse_args ()
    model =args .judge_model .strip ()
    endpoint =args .judge_endpoint .strip ()
    if not model or not endpoint :
        parser .error ("--judge-model and --judge-endpoint must be nonempty")
    key =os .environ .get ("JUDGE_API_KEY")
    if not key :
        parser .error ("JUDGE_API_KEY is required")
    options =json .loads (args .judge_options )
    if not isinstance (options ,dict ):
        parser .error ("--judge-options must be a JSON object")

    def complete (image_url ,prompt ):
        return completion (endpoint ,model ,key ,image_url ,prompt ,parameters ={"max_tokens":8192 ,**options})

    evaluate (args .data .resolve (),args .predictions .resolve (),args .output .resolve (),complete )


if __name__ =="__main__":
    main ()
