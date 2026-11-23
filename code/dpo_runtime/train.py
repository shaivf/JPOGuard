from __future__ import annotations 

import argparse 
import hashlib 
import json 
import random 
import sys 
from collections import defaultdict 
from pathlib import Path 

from .data import append_jsonl ,load_unique_by_id ,pair_digest ,read_json ,validate_partition 
from .prompt import joint_prompt ,JOINT_PROMPT 
from .length_policy import validate_pair_lengths 


def sha256_file (path :Path )->str :
    digest =hashlib .sha256 ()
    with open (path ,"rb")as handle :
        for block in iter (lambda :handle .read (1024 *1024 ),b""):
            digest .update (block )
    return digest .hexdigest ()


def dpo_logit (policy_chosen ,policy_rejected ,reference_chosen ,reference_rejected ):
    ''
    return (policy_chosen -policy_rejected )-(reference_chosen -reference_rejected )


ENCODING_VERSION ="correction_only_full_response_v1"


def response_texts (pair ):
    ''
    return (json .dumps (pair ["chosen_response"],ensure_ascii =False ,indent =4 ),
    json .dumps (pair ["rejected_response"],ensure_ascii =False ,indent =4 ))


def load_checked_pairs (args ):
    source =args .data .resolve ()
    train =read_json (args .train_data or source /"train_data.json")
    test =read_json (args .test_data or source /"test_data.json")
    correction =read_json (args .correction_data or source /"test_correction.json")
    pairs =load_unique_by_id (args .pairs )
    if args .limit :
        pairs =dict (list (pairs .items ())[:args .limit ])
    validate_partition (set (pairs ),train ,test ,correction )
    policy_file =args .pairs .with_suffix (args .pairs .suffix +".length_policy.json")
    if not policy_file .is_file ():
        raise ValueError ("New DPO pairs require new-SFT candidate provenance; old pairs are audit-only")
    policy =read_json (policy_file )
    reference_adapter =args .reference_adapter if args .mode =="evaluate"else args .adapter 
    actual_sft =sha256_file (Path (reference_adapter )/"adapter_model.safetensors")
    dataset_sft =policy .get ("new_sft_adapter_sha256")
    if (policy .get ("extra_words")!=3 
    or policy .get ("pairs_sha256")!=sha256_file (args .pairs )
    or policy .get ("joint_template_sha256")!=hashlib .sha256 (JOINT_PROMPT .encode ()).hexdigest ()
    or (not args .use_released_pairs and
    (not dataset_sft or policy .get ("candidate_adapter_sha256")!=dataset_sft
    or dataset_sft !=actual_sft))
    or policy .get ("fresh_candidates_verified")is not True ):
        raise ValueError ("New DPO pair/model/prompt provenance does not match")
    expected_split ="train"if args .mode =="train"else args .pair_split 
    for sample_id ,pair in pairs .items ():
        if pair .get ("split")!=expected_split :
            raise ValueError (
            f"DPO {args .mode } file contains non-{expected_split } ID: {sample_id }"
            )
        validate_pair_lengths (train [sample_id ],pair )
        if pair .get ("pair_digest")!=pair_digest (pair ):
            raise ValueError (f"pair digest mismatch: {sample_id }")
        chosen_response =pair ["chosen_response"]
        rejected_response =pair ["rejected_response"]
        if (not chosen_response .get ("Reason")
        or rejected_response .get ("Reason")!=chosen_response ["Reason"]
        or chosen_response ==rejected_response ):
            raise ValueError (f"invalid hybrid response pair: {sample_id }")
        kind =pair .get ("pair_type")
        if kind not in ("rewrite_error","original_fallback"):
            raise ValueError (f"unknown hybrid pair type: {sample_id } {kind }")
        if (kind =="original_fallback"and 
        rejected_response ["Rewritten_Headline"].strip ()!=
        str (train [sample_id ].get ("caption")or "").strip ()):
            raise ValueError (f"fallback does not match original headline: {sample_id }")
    images =args .images or source /"train_img"
    return train ,pairs ,images 


def make_model (args ,trainable :bool ):
    import torch 
    from peft import PeftModel ,prepare_model_for_kbit_training 
    from transformers import AutoModelForImageTextToText ,AutoProcessor ,AutoTokenizer ,BitsAndBytesConfig 

    tokenizer =AutoTokenizer .from_pretrained (args .base_model ,use_fast =False ,trust_remote_code =True )
    processor =AutoProcessor .from_pretrained (args .base_model )
    quant =BitsAndBytesConfig (
    load_in_4bit =True ,bnb_4bit_quant_type ="nf4",
    bnb_4bit_compute_dtype =torch .bfloat16 ,bnb_4bit_use_double_quant =True ,
    )
    base =AutoModelForImageTextToText .from_pretrained (
    args .base_model ,quantization_config =quant ,device_map ="auto",
    torch_dtype =torch .bfloat16 ,trust_remote_code =True ,
    )
    base =prepare_model_for_kbit_training (base ,use_gradient_checkpointing =trainable )
    model =PeftModel .from_pretrained (base ,args .adapter ,is_trainable =trainable )
    if trainable :
        model .enable_input_require_grads ()
        model .train ()


        for module in model .modules ():
            if isinstance (module ,torch .nn .Dropout ):
                module .eval ()
    else :
        model .eval ()
    return model ,processor ,tokenizer 


def encoded_pair (model ,processor ,tokenizer ,train_rec ,pair ,image_path ):
    from dpo_train import encode 

    prompt =joint_prompt (train_rec )
    chosen_text ,rejected_text =response_texts (pair )
    chosen =encode (model ,processor ,tokenizer ,prompt ,str (image_path ),chosen_text )
    rejected =encode (model ,processor ,tokenizer ,prompt ,str (image_path ),rejected_text )
    return chosen ,rejected 


def logp (model ,encoded ):
    from dpo_train import compute_logps 
    return compute_logps (model ,*encoded ,model .device )


def reference (args ,train ,pairs ,images ):
    import torch 

    output =args .out 
    if output .exists ()and not args .resume :
        raise FileExistsError (f"{output } exists; use --resume or a new output path")
    meta_path =output .with_suffix (output .suffix +".meta.json")
    from .reference_binding import expected_meta ,validate_meta ,validate_rows 
    if args .resume and output .exists ():
        if not meta_path .exists ():
            raise ValueError ("Reference metadata missing; use a new output path")
        saved_meta =read_json (meta_path )
        if saved_meta .get ("schema_version")!=2 :
            raise ValueError ("Legacy reference cache has no input binding; preserve it and use a new output path")
    meta =expected_meta (args ,train ,images ,args .adapter ,ENCODING_VERSION ,response_texts )
    if args .resume and output .exists ():
        validate_meta (read_json (meta_path ),meta )
        existing =load_unique_by_id (output )
        if not set (existing )<=set (pairs ):
            raise ValueError ("reference cache contains IDs outside current pairs")
        for sample_id ,row in existing .items ():
            if row .get ("pair_digest")!=pairs [sample_id ]["pair_digest"]:
                raise ValueError (f"reference cache pair changed: {sample_id }")
        validate_rows (existing ,load_unique_by_id (args .pairs ),meta )
        done =set (existing )
    else :
        done =set ()
    items =[(sample_id ,pair )for sample_id ,pair in pairs .items ()if sample_id not in done ]
    if not items :
        print ("[reference] no remaining pairs")
        return 
    output .parent .mkdir (parents =True ,exist_ok =True )
    meta_path .write_text (json .dumps (meta ,ensure_ascii =False ,indent =2 )+"\n",encoding ="utf-8")
    model ,processor ,tokenizer =make_model (args ,trainable =False )
    with torch .no_grad ():
        for number ,(sample_id ,pair )in enumerate (items ,1 ):
            image_path =images /f"{sample_id }.jpg"
            if not image_path .is_file ():
                raise FileNotFoundError (image_path )
            chosen ,rejected =encoded_pair (
            model ,processor ,tokenizer ,train [sample_id ],pair ,image_path 
            )
            row ={"id":sample_id ,"pair_digest":pair ["pair_digest"],
            "input_digest":meta ["input_binding"]["input_digests"][sample_id ],
            "ref_chosen_logp":float (logp (model ,chosen ).item ()),
            "ref_rejected_logp":float (logp (model ,rejected ).item ())}
            append_jsonl (output ,row )
            if number %20 ==0 or number ==len (items ):
                print (f"[reference] {number }/{len (items )}")


def train_dpo (args ,train ,pairs ,images ):
    import torch 
    import torch .nn .functional as F 

    refs =load_unique_by_id (args .reference )
    if not set (pairs )<=set (refs ):
        raise ValueError (f"reference coverage mismatch: pairs={len (pairs )}, refs={len (refs )}")
    meta =read_json (args .reference .with_suffix (args .reference .suffix +".meta.json"))
    if meta .get ("reference")!="base_plus_frozen_sft_adapter":
        raise ValueError ("reference was not computed from the frozen SFT model")
    if meta .get ("encoding")!=ENCODING_VERSION :
        raise ValueError ("reference response encoding differs from policy")
    if meta .get ("adapter_sha256")!=sha256_file (Path (args .adapter )/"adapter_model.safetensors"):
        raise ValueError ("reference adapter hash differs from policy starting adapter")
    if meta .get ("base_model")!=str (args .base_model ):
        raise ValueError ("reference base model path differs")
    for sample_id ,pair in pairs .items ():
        if refs [sample_id ].get ("pair_digest")!=pair ["pair_digest"]:
            raise ValueError (f"reference pair digest mismatch: {sample_id }")

    validate_reference (args ,train ,pairs ,images ,args .adapter )
    if args .out .exists ():
        raise FileExistsError (args .out )

    torch .manual_seed (args .seed )
    model ,processor ,tokenizer =make_model (args ,trainable =True )
    params =[parameter for parameter in model .parameters ()if parameter .requires_grad ]
    if not params :
        raise RuntimeError ("no trainable adapter parameters")
    optimizer =torch .optim .AdamW (params ,lr =args .lr )
    args .out .mkdir (parents =True ,exist_ok =False )
    (args .out /"config.json").write_text (json .dumps ({
    "source_adapter":str (args .adapter ),"reference":str (args .reference ),
    "pairs":str (args .pairs ),"beta":args .beta ,"lr":args .lr ,
    "epochs":args .epochs ,"seed":args .seed ,"reference_kind":meta ["reference"],
    },ensure_ascii =False ,indent =2 )+"\n",encoding ="utf-8")
    metrics_path =args .out /"metrics.jsonl"
    step =0 
    for epoch in range (args .epochs ):
        order =list (pairs )
        random .Random (args .seed +epoch ).shuffle (order )
        for number ,sample_id in enumerate (order ,1 ):
            pair =pairs [sample_id ]
            image_path =images /f"{sample_id }.jpg"
            if not image_path .is_file ():
                raise FileNotFoundError (image_path )
            chosen ,rejected =encoded_pair (
            model ,processor ,tokenizer ,train [sample_id ],pair ,image_path 
            )
            policy_chosen =logp (model ,chosen )
            policy_rejected =logp (model ,rejected )
            margin =dpo_logit (
            policy_chosen ,policy_rejected ,
            refs [sample_id ]["ref_chosen_logp"],
            refs [sample_id ]["ref_rejected_logp"],
            )
            loss =-F .logsigmoid (args .beta *margin )
            optimizer .zero_grad (set_to_none =True )
            loss .backward ()
            gradients =[parameter .grad for parameter in params if parameter .grad is not None ]
            if not gradients or not any (torch .count_nonzero (grad ).item ()for grad in gradients ):
                raise RuntimeError (f"no nonzero LoRA gradients: {sample_id }")
            grad_norm =torch .nn .utils .clip_grad_norm_ (params ,1.0 )
            if not torch .isfinite (grad_norm ):
                raise RuntimeError (f"non-finite LoRA gradient norm: {sample_id }")
            optimizer .step ()
            step +=1 
            append_jsonl (metrics_path ,{
            "epoch":epoch ,"row":number ,"step":step ,"id":sample_id ,
            "pair_type":pair .get ("pair_type","legacy"),
            "loss":float (loss .item ()),"margin":float (margin .item ()),
            "grad_norm":float (grad_norm .item ()),
            })
            if step %args .log_steps ==0 or step ==1 :
                print (f"[dpo] epoch={epoch } row={number }/{len (order )} step={step } "
                f"type={pair .get ('pair_type','legacy')} "
                f"loss={loss .item ():.5f} margin={margin .item ():+.3f} "
                f"grad_norm={grad_norm .item ():.4f}")
            if step %args .save_steps ==0 :
                model .save_pretrained (args .out /f"checkpoint-{step }")
    model .save_pretrained (args .out /"final")
    print (f"[dpo] completed steps={step } final={args .out /'final'}")


def evaluate_dpo (args ,train ,pairs ,images ):
    import torch 
    import torch .nn .functional as F 

    refs =load_unique_by_id (args .reference )
    if set (refs )!=set (pairs ):
        raise ValueError (f"evaluation reference coverage mismatch: pairs={len (pairs )}, refs={len (refs )}")
    meta =read_json (args .reference .with_suffix (args .reference .suffix +".meta.json"))
    if meta .get ("reference")!="base_plus_frozen_sft_adapter":
        raise ValueError ("evaluation reference is not the frozen SFT policy")
    if meta .get ("encoding")!=ENCODING_VERSION :
        raise ValueError ("evaluation reference response encoding differs from policy")
    if meta .get ("base_model")!=str (args .base_model ):
        raise ValueError ("evaluation reference base model differs")
    if not args .reference_adapter :
        raise ValueError ("--reference-adapter is required for evaluation")
    reference_hash =sha256_file (Path (args .reference_adapter )/"adapter_model.safetensors")
    if meta .get ("adapter_sha256")!=reference_hash :
        raise ValueError ("evaluation reference adapter hash differs from frozen SFT adapter")
    for sample_id ,pair in pairs .items ():
        if refs [sample_id ].get ("pair_digest")!=pair ["pair_digest"]:
            raise ValueError (f"evaluation reference pair digest mismatch: {sample_id }")

    validate_reference (args ,train ,pairs ,images ,args .reference_adapter )

    model ,processor ,tokenizer =make_model (args ,trainable =False )
    records =[]
    grouped =defaultdict (list )
    with torch .no_grad ():
        for number ,(sample_id ,pair )in enumerate (pairs .items (),1 ):
            image_path =images /f"{sample_id }.jpg"
            if not image_path .is_file ():
                raise FileNotFoundError (image_path )
            chosen ,rejected =encoded_pair (
            model ,processor ,tokenizer ,train [sample_id ],pair ,image_path 
            )
            policy_chosen =logp (model ,chosen )
            policy_rejected =logp (model ,rejected )
            margin =dpo_logit (
            policy_chosen ,policy_rejected ,
            refs [sample_id ]["ref_chosen_logp"],
            refs [sample_id ]["ref_rejected_logp"],
            )
            loss =-F .logsigmoid (args .beta *margin )
            record ={
            "id":sample_id ,"pair_type":pair .get ("pair_type","legacy"),
            "loss":float (loss .item ()),"margin":float (margin .item ()),
            "preferred":bool (margin .item ()>0 ),
            }
            records .append (record )
            grouped [record ["pair_type"]].append (record )
            if number %args .log_steps ==0 or number ==len (pairs ):
                print (f"[eval] {number }/{len (pairs )}")

    def summarize (rows ):
        return {
        "count":len (rows ),
        "mean_loss":sum (row ["loss"]for row in rows )/len (rows ),
        "mean_margin":sum (row ["margin"]for row in rows )/len (rows ),
        "preference_accuracy":sum (row ["preferred"]for row in rows )/len (rows ),
        }

    result ={
    "policy_adapter":str (args .adapter ),
    "policy_adapter_sha256":sha256_file (Path (args .adapter )/"adapter_model.safetensors"),
    "reference_adapter":str (args .reference_adapter ),
    "reference_adapter_sha256":reference_hash ,
    "pairs":str (args .pairs ),"reference":str (args .reference ),
    "beta":args .beta ,"split":args .pair_split ,
    "overall":summarize (records ),
    "by_type":{name :summarize (rows )for name ,rows in sorted (grouped .items ())},
    "records":records ,
    }
    if args .out .exists ():
        raise FileExistsError (args .out )
    args .out .parent .mkdir (parents =True ,exist_ok =True )
    args .out .write_text (json .dumps (result ,ensure_ascii =False ,indent =2 )+"\n",encoding ="utf-8")
    print (json .dumps ({"overall":result ["overall"],"by_type":result ["by_type"]},ensure_ascii =False ))


def validate_reference (args ,train ,pairs ,images ,adapter ):
    from .reference_binding import expected_meta ,validate_meta ,validate_rows 
    meta =read_json (args .reference .with_suffix (args .reference .suffix +".meta.json"))
    if meta .get ("schema_version")!=2 :
        raise ValueError ("Legacy reference cache has no input binding; preserve it and calculate a new cache at a new output path")
    expected =expected_meta (args ,train ,images ,adapter ,ENCODING_VERSION ,response_texts )
    validate_meta (meta ,expected )
    rows =load_unique_by_id (args .reference )
    if not set (pairs )<=set (rows ):
        raise ValueError ("Reference cache does not cover requested pairs")
    validate_rows (rows ,load_unique_by_id (args .pairs ),meta )
    return rows 


def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("mode",choices =("reference","train","evaluate"))
    parser .add_argument ("--base-model",required =True )
    parser .add_argument ("--adapter",required =True )
    parser .add_argument ("--use-released-pairs",action ="store_true")
    parser .add_argument ("--data",type =Path ,required =True )
    parser .add_argument ("--src-fixed",type =Path ,default =Path (__file__ ).resolve ().parents [1 ]/"model")
    parser .add_argument ("--train-data",type =Path )
    parser .add_argument ("--test-data",type =Path )
    parser .add_argument ("--correction-data",type =Path )
    parser .add_argument ("--images",type =Path )
    parser .add_argument ("--pairs",type =Path ,required =True )
    parser .add_argument ("--reference",type =Path )
    parser .add_argument ("--reference-adapter",type =Path ,
    help ="Frozen SFT adapter used to create the reference cache")
    parser .add_argument ("--out",type =Path ,required =True )
    parser .add_argument ("--epochs",type =int ,default =1 )
    parser .add_argument ("--lr",type =float ,default =5e-6 )
    parser .add_argument ("--beta",type =float ,default =0.1 )
    parser .add_argument ("--seed",type =int ,default =42 )
    parser .add_argument ("--limit",type =int ,default =0 )
    parser .add_argument ("--save-steps",type =int ,default =100 )
    parser .add_argument ("--log-steps",type =int ,default =10 )
    parser .add_argument ("--resume",action ="store_true",help ="Resume reference calculation only")
    parser .add_argument ("--check-reference-only",action ="store_true",help ="Validate reference provenance without loading model weights")
    parser .add_argument ("--pair-split",choices =("train","dev"),default ="train",
    help ="Expected pair split for reference/evaluate; train mode always requires train")
    args =parser .parse_args ()
    if args .resume and args .mode !="reference":
        parser .error ("--resume applies only to reference calculation")
    if args .check_reference_only and not args .reference :
        parser .error ("--check-reference-only requires --reference")
    if args .epochs <1 or args .beta <=0 or args .lr <=0 :
        parser .error ("epochs, beta and lr must be positive")
    if args .mode in ("train","evaluate")and not args .reference :
        parser .error ("--reference is required for training/evaluation")
    if args .mode =="evaluate"and not args .reference_adapter :
        parser .error ("--reference-adapter is required for evaluation")
    sys .path .insert (0 ,str (args .src_fixed ))
    train ,pairs ,images =load_checked_pairs (args )
    print (f"[dpo] clean train pairs={len (pairs )}")
    if args .check_reference_only :
        validate_reference (args ,train ,pairs ,images ,args .reference_adapter or args .adapter )
        print ("[reference] complete input binding validated")
        return 
    if args .mode =="reference":
        reference (args ,train ,pairs ,images )
    elif args .mode =="train":
        train_dpo (args ,train ,pairs ,images )
    else :
        evaluate_dpo (args ,train ,pairs ,images )


if __name__ =="__main__":
    main ()
