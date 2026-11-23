import argparse 
from pathlib import Path 

from evaluate import evaluate ,test_inputs 
from train_dpo import GROUPS ,sft_adapter ,train 


def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("variant",choices =("joint_sft",*GROUPS [1 :]))
    parser .add_argument ("--base-model",type =Path ,required =True )
    parser .add_argument ("--sft-adapter",type =Path ,required =True )
    parser .add_argument ("--output",type =Path ,required =True )
    parser .add_argument ("--data",type =Path ,required =True )
    parser .add_argument ("--pairs-dir",type =Path )
    parser .add_argument ("--reader-data",type =Path ,required =True )
    args =parser .parse_args ()
    if args .variant !="joint_sft"and args .pairs_dir is None :
        parser .error ("--pairs-dir is required for DPO variants")
    base =args .base_model .resolve ()
    output =args .output .resolve ()
    data =args .data .resolve ()
    test_set =test_inputs (data ,args .reader_data .resolve ())
    adapter =sft_adapter (args .sft_adapter )
    if args .variant =="joint_sft":
        evaluate (test_set ,base ,adapter ,output )
    else :
        selected =train (data ,args .pairs_dir .resolve (),base ,adapter ,output ,args .variant )
        evaluate (test_set ,base ,sft_adapter (selected ),output /"official_test")


if __name__ =="__main__":
    main ()
