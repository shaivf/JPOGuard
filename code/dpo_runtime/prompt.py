from pathlib import Path 

from .length_policy import render_joint 

JOINT_PROMPT =(Path (__file__ ).resolve ().parents [1 ]/"prompts/joint_template.txt").read_text (encoding ="utf-8")


def joint_prompt (record ):
    return render_joint (record ,JOINT_PROMPT )
