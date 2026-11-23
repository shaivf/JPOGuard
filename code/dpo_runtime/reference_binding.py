''
from functools import lru_cache 
from importlib .metadata import version ,distribution ,PackageNotFoundError 
from pathlib import Path 
import hashlib 
import json 
import math 

from .data import load_unique_by_id ,pair_digest ,read_json 
from .prompt import joint_prompt 


def sha (path ):
    d =hashlib .sha256 ()
    with Path (path ).open ('rb')as f :
        for block in iter (lambda :f .read (1024 *1024 ),b''):
            d .update (block )
    return d .hexdigest ()


def digest (value ):
    return hashlib .sha256 (json .dumps (value ,ensure_ascii =False ,sort_keys =True ,separators =(',',':')).encode ()).hexdigest ()


@lru_cache (maxsize =128 )
def asset_sha (path ,size ,mtime_ns ):
    return sha (path )


def base_assets (path ):
    path =Path (path ).resolve ()
    if not path .is_dir ():
        raise ValueError ('Reference binding requires local model assets with a pinned revision')
    result ={}
    for p in sorted (path .iterdir ()):
        if p .is_file ()and p .suffix in ('.json','.txt','.model','.tiktoken','.safetensors','.bin','.jinja','.py'):
            s =p .stat ()
            result [p .name ]=asset_sha (str (p ),s .st_size ,s .st_mtime_ns )
    if not result :
        raise ValueError ('No model assets found for reference binding')
    return result 


def expected_meta (args ,train ,images ,adapter ,encoding ,response_texts ):
    source =Path (args .src_fixed )
    full_pairs =load_unique_by_id (args .pairs )
    identities ={}
    for sid ,pair in full_pairs .items ():
        if pair .get ('pair_digest')!=pair_digest (pair ):
            raise ValueError (f'Reference binding pair digest mismatch: {sid }')
        image =Path (images )/f'{sid }.jpg'
        if not image .is_file ():
            raise FileNotFoundError (image )
        chosen ,rejected =response_texts (pair )
        identities [sid ]=digest ({'prompt':joint_prompt (train [sid ]),'image_sha256':sha (image ),
        'chosen_text':chosen ,'rejected_text':rejected ,'pair_digest':pair ['pair_digest']})
    versions ={}
    for package in ('torch','transformers','tokenizers','peft','qwen-vl-utils','Pillow','bitsandbytes'):
        try :versions [package ]=version (package )
        except PackageNotFoundError :versions [package ]=None 
    model_type =read_json (Path (args .base_model )/'config.json')['model_type']
    model_sources =(f"transformers/models/{model_type .replace ('-','_')}/",
    'transformers/models/auto/','qwen_vl_utils/')
    processing_files ={'transformers/processing_utils.py','transformers/image_processing_utils.py',
    'transformers/tokenization_utils_base.py','transformers/video_processing_utils.py'}
    library_sources ={}
    for package in ('transformers','qwen-vl-utils'):
        try :dist =distribution (package )
        except PackageNotFoundError :continue 
        for entry in dist .files or ():
            name =str (entry ).replace ('\\','/')
            processor_source =name .startswith ('transformers/models/')and Path (name ).name .startswith (
            ('image_processing_','video_processing_'))
            if name .endswith ('.py')and (name .startswith (model_sources )or name in processing_files or processor_source ):
                path =Path (dist .locate_file (entry ))
                if path .is_file ():library_sources [name ]=sha (path )
    binding ={'input_digests':identities ,'base_assets':base_assets (args .base_model ),
    'encoding_source_sha256':{'dpo_train.py':sha (source /'dpo_train.py'),
    'common.py':sha (source /'common.py'),
    'prompt.py':sha (Path (__file__ ).with_name ('prompt.py')),
    'train.py':sha (Path (__file__ ).with_name ('train.py')),
    'reference_binding.py':sha (__file__ )},
    'resized':280 ,'max_length':8192 ,'package_versions':versions ,'library_source_sha256':library_sources }
    return {'schema_version':2 ,'reference':'base_plus_frozen_sft_adapter','encoding':encoding ,
    'adapter_sha256':sha (Path (adapter )/'adapter_model.safetensors'),
    'base_model':str (args .base_model ),'pairs':str (args .pairs ),
    'input_binding':binding ,'binding_digest':digest (binding )}


def validate_meta (saved ,expected ):
    if saved .get ('schema_version')!=2 or 'input_binding'not in saved :
        raise ValueError ('Legacy reference cache has no complete input binding; preserve it and calculate a new cache at a new output path')
    if saved !=expected :
        raise ValueError ('Reference inputs, image, responses, encoding, model assets or environment changed; cannot reuse cache')


def validate_rows (rows ,pairs ,meta ):
    for sid ,row in rows .items ():
        if sid not in pairs :
            raise ValueError (f'Reference ID outside current pair file: {sid }')
        if row .get ('pair_digest')!=pairs [sid ]['pair_digest']:
            raise ValueError (f'Reference pair changed: {sid }')
        if row .get ('input_digest')!=meta ['input_binding']['input_digests'][sid ]:
            raise ValueError (f'Reference input binding mismatch: {sid }')
        for key in ('ref_chosen_logp','ref_rejected_logp'):
            value =row .get (key )
            if isinstance (value ,bool )or not isinstance (value ,(int ,float ))or not math .isfinite (value ):
                raise ValueError (f'Invalid reference score {key }: {sid }')
