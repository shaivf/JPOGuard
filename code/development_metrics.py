import hashlib 
import json 
import re 

LABEL =re .compile (r'"Misleading"\s*:\s*"(Yes|No)"',re .I )

def digest (value ):
    return hashlib .sha256 (json .dumps (value ,ensure_ascii =False ,sort_keys =True ,separators =(',',':')).encode ()).hexdigest ()


def label_prefix (raw ):
    matches =LABEL .findall (raw )
    if len (matches )!=1 :
        raise ValueError ('Expected one complete Misleading label in generated prefix')
    return matches [0 ].capitalize ()
