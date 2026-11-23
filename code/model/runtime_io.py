def parse_correction_response (raw ,extractor ,resolver ):
    base ={'raw':raw ,'raw_len':len (raw ),'pred_label':'','rewritten_headline':''}
    obj =extractor (raw )
    if obj is None :
        return dict (base ,status ='parse_error',error_type ='json_parse_error',parse_error ='No valid JSON object in model response')
    label =resolver (obj ,'Misleading',default =None )
    headline =resolver (obj ,'Rewritten_Headline','Rewriten_Headline',default =None )
    reason =resolver (obj ,'Reason',default =None )
    if not isinstance (label ,str )or label .strip ().lower ()not in ('yes','no')or not isinstance (headline ,str )or not isinstance (reason ,str ):
        return dict (base ,status ='parse_error',error_type ='response_schema_error',parse_error ='Expected Yes/No label and string Reason/Rewritten_Headline',parsed_response =obj )
    label ,headline =label .strip ().capitalize (),headline .strip ()
    outcome ='rewritten'if headline else 'model_no_rewrite'if label =='No'else 'model_empty_rewrite'
    return dict (base ,status ='ok',pred_label =label ,rewritten_headline =headline ,reason =reason ,outcome =outcome )
