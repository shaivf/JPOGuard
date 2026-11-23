import json 
from pathlib import Path 

from dpo_runtime .data import load_unique_by_id ,read_json 


SPLITS =Path (__file__ ).resolve ().parent /"splits"


def training_splits (news ):
    sft =read_json (SPLITS /"sft_splits.json")
    hard =read_json (SPLITS /"hard_pool_splits.json")
    train =sft ["sft_train_ids"]
    validation =sft ["sft_validation_ids"]
    hard_ids ={sid :split for split ,ids in hard .items ()for sid in ids }
    if ((len (train ),len (validation ),len (hard ["train"]),len (hard ["dev"]))!=(4109 ,457 ,391 ,43 )
    or len (hard_ids )!=434 or len (set (train +validation ))!=4566 
    or (set (train +validation )&hard_ids .keys ())
    or (set (train +validation )|hard_ids .keys ())!=news .keys ()):
        raise ValueError ("Official training IDs do not match the experiment splits")
    return train ,validation ,hard_ids 


def annotation (record ):
    value =record ["Misleading"]
    return json .loads (value )if isinstance (value ,str )else value 


def append_session (output ,resume ):
    output .parent .mkdir (parents =True ,exist_ok =True )
    if resume and output .is_file ():
        return load_unique_by_id (output )
    output .touch (exist_ok =False )
    return {}
