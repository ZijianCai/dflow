import json
from typing import List
from dflow import (
    Workflow,
    Step,
    argo_range,
    SlurmRemoteExecutor,
    upload_artifact,
    download_artifact,
    InputArtifact,
    OutputArtifact,
    ShellOPTemplate
)
from dflow.python import (
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    Slices
)
import subprocess, os, shutil, glob
from pathlib import Path
from typing import List


class Screen(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'dirct': str,
            'api': str,
            'screen_code': Artifact(Path),
            'atom_init': Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'DownloadCIF_output' : Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        #
        cmd = f'python {op_in["screen_code"]} {op_in["api"]} {op_in["dirct"]}'
        subprocess.call(cmd, shell=True)
        #copy atom_init.json file to the results folder for cgcnn
        cmd = f'cp {op_in["atom_init"]} {op_in["dirct"]}'
        subprocess.call(cmd, shell=True)
        return OPIO({
            "DownloadCIF_output": Path(f'{op_in["dirct"]}'), 
        })

class Train(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            #'train_parameter': list,
            'train_code': Artifact(Path),
            'train_data_set': Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'trained_model' : Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        cmd = f'cd {op_in["train_code"]} && \
        python main.py \
        --train-size 50 --val-size 2 --test-size 1 \
        {op_in["train_data_set"]}'
        subprocess.call(cmd, shell=True)
        return OPIO({
            "trained_model": Path(op_in["train_code"])/'model_best.pth.tar'
        })

class Predict(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'predict_code': Artifact(Path),
            'predict_model': Artifact(Path),
            'predict_data_set': Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'predict_result' : Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        cmd = f'cd {op_in["predict_code"]} && \
        python predict.py {op_in["predict_model"]} {op_in["predict_data_set"]}'
        subprocess.call(cmd, shell=True)

        return OPIO({
            "predict_result": Path(op_in["predict_code"])/'test_results.csv' 
        })




def main():
    screen_name_ls = ["screen_1.py", "screen_2.py"] #name of screening codes
    screen_list = []    #uploade screening codes
    step_screen_ls = []  #screening steps list
    atom_init = upload_artifact("atom_init.json")   #json file
    train_main = upload_artifact("train")           #traning code
    predict_main = upload_artifact("predict")       #prediction code

    for i, criteria in enumerate(screen_name_ls):
        screen_list.append(upload_artifact(screen_name_ls[i]))    #uploade screening codes
        step_screen =  Step(
            f"screen-{i}",
            PythonOPTemplate(Screen, image = '-- your image --',),
            artifacts = {"atom_init": atom_init, \
                        "screen_code": screen_list[i]},
            parameters = {"dirct": screen_name_ls[i][:-3], "api": '-- your api-key from materialsproject --'},
        )
        step_screen_ls.append(step_screen)

    train = Step(
        "train",
        PythonOPTemplate(Train, image = '-- your image --',),
        artifacts={"train_code": train_main, \
                    "train_data_set": step_train_ls[0].outputs.artifacts["DownloadCIF_output"],},
    )

    predict = Step(
        "predict",
        PythonOPTemplate(Predict, image = '-- your image --',),
        artifacts={"predict_code": predict_main, \
                    "predict_data_set": step_train_ls[1].outputs.artifacts["DownloadCIF_output"],\
                    "predict_model": train.outputs.artifacts["trained_model"],}, #use the second 
    )


    wf = Workflow("screenning-ML")
    # wf.add(screen_1)
    wf.add(step_screen_ls)
    wf.add(train)
    wf.add(predict)
    wf.submit()

if __name__ == '__main__':

    main()
