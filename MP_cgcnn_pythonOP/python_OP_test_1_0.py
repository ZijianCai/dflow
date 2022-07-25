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


from dflow.python import (
    OP,
    OPIO,
    OPIOSign
    )


class Screen(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            'dirct': str,
            'api': str,
            'screen_code': Artifact(Path)
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            'DownloadCIF_output' : Artifact(Path),
            'excel_results': Artifact(Path),
        })

    @OP.exec_sign_check
    def execute(self, op_in: OPIO) -> OPIO:
        # for ii in range(op_in["repeat_number"]):
        #     print("hello world")
        cmd = f'python {op_in["screen_code"]} {op_in["api"]} {op_in["dirct"]}'
        subprocess.call(cmd, shell=True)
        return OPIO({
            "DownloadCIF_output": Path(f'{op_in["dirct"]}'), 
            "excel_results": Path('id_prop.csv')
        })



def main():
    screen_name_ls = ["screen_1.py", "screen_2.py"]
    #atom_init = upload_artifact("atom_init.json")
    artifact0 = upload_artifact("screen_1.py")
    artifact1 = upload_artifact("screen_2.py")
    #train_main = upload_artifact("main.py")
    step_train_ls = []

    # screen_1 = Step(
    #     "screen1",
    #     PythonOPTemplate(Screen, image = 'kianpu/cgcnn:v1.1.0',),
    #     artifacts={"screen_code": artifact1},
    #     parameters={"dirct": 'results1', "api": 'v8mU74QhZSIisN26'},
    # )
    for i, criteria in enumerate(screen_name_ls):
        step_screen =  Step(
            f"screen-{i}",
            PythonOPTemplate(Screen, image = 'kianpu/cgcnn:v1.1.0',),
            artifacts = {"screen_code": f'artifact{i}'},
            parameters = {"dirct": f'results{i}', "api": 'v8mU74QhZSIisN26'},
        )
        step_train_ls.append(step_screen)


    wf = Workflow("print-hello")
    # wf.add(screen_1)
    wf.add(step_train_ls)
    wf.submit()

if __name__ == '__main__':

    main()


"""
from dflow import ShellOPTemplate, InputParameter, InputArtifact, OutputParameter, OutputArtifact
from dflow import Workflow
from dflow import upload_artifact
from dflow import Step

MP_id = "v8mU74QhZSIisN26"
screen_name_ls = ["/screen_artifact/screen_1.py", "/screen_artifact/screen_2.py"]
results_path_ls = ["/screen_1", "/screen_2"]


screening = ShellOPTemplate(
    name = 'screening',
    image = 'kianpu/cgcnn:latest',
    script = ". ~/.bashrc && python {{inputs.parameters.screen_name}} {{inputs.parameters.MP_id}} {{inputs.parameters.results_path}} && ls"
)


screen_artifact = upload_artifact(["screen_1.py", "screen_2.py"])
screening.inputs.artifacts = {"screen_inp_art": InputArtifact(path="screen_artifact")}
screening.inputs.parameters = {"MP_id": InputParameter(),
                                "screen_name": InputParameter(),
                                "results_path": InputParameter()}
screening.outputs.artifacts = {"screen_out_art": OutputArtifact(path = "/screen")}

step_train_ls = []

for i, criteria in enumerate(screen_name_ls):
    step_screen =  Step(
        #name = screening,
        name = f"screen-{i}",
        template = screening,
        artifacts = {"screen_inp_art": screen_artifact},
        parameters = {"MP_id" : MP_id,
                    "screen_name": screen_name_ls[i],
                    "results_path": results_path_ls[i]},
    )
    step_train_ls.append(step_screen)


wf = Workflow(name="screening")
wf.add(step_train_ls)
#wf.add(step_screen)
wf.submit()

"""

"""
train = ShellOPTemplate(
    name="train",
    image="kianpu/cgcnn:latest",
    script=". ~/.bashrc && python /train_artifact/main.py --train-size 500 --val-size 20 --test-size 10 \
            /train_artifact/root_dir"
    # Will generate /model_best.pth.tar; checkpoint.pth.tar; test_results.csv
    # /predict_artifact/predict.py \
            #/predict_artifact/model_best.pth.tar \
            #/predict_artifact/data_to_be_predicted/sample-regression \
            
)

screening = ShellOPTemplate(
    name = 'screening',
    image = 'kianpu/cgcnn:latest',
    script = ''
)

predict = ShellOPTemplate(
    name="predict",
    image="kianpu/cgcnn:latest",
    script="cp /model_best.pth.tar /predict_artifact/model_best.pth.tar && . ~/.bashrc && \
            python /predict_artifact/predict.py \
            /predict_artifact/model_best.pth.tar \
            /predict_artifact/data_to_be_predicted/sample-regression \
            "
)


from dflow import upload_artifact
# define input

train_artifact = upload_artifact(["cgcnn", "root_dir", "main.py"])
predict_artifact = upload_artifact(["predict.py","data_to_be_predicted","cgcnn"])
#simple_example_templ.inputs.parameters = {"msg": InputParameter()}
predict.inputs.artifacts = {"predict_inp_art": InputArtifact(path="predict_artifact"), \
                            "predict_model": InputArtifact(path="/model_best.pth.tar")}
train.inputs.artifacts = {"train_inp_art": InputArtifact(path="train_artifact")}
# define output
train.outputs.artifacts = {"out_art_model": OutputArtifact(path="model_best.pth.tar")}
predict.outputs.artifacts = {"out_art": OutputArtifact(path="test_results.csv")}

from dflow import Step

train_step = Step(
    name="train",
    template=train,
    #parameters={"msg": "HelloWorld!"},
    artifacts={"train_inp_art": train_artifact},
)


predict_step = Step(
    name="predict",
    template=predict,
    #parameters={"msg": "HelloWorld!"},
    artifacts={"predict_inp_art": predict_artifact, \
                "predict_model": train_step.outputs.artifacts["out_art_model"] \
                },
)


from dflow import Workflow

wf = Workflow(name="train")
wf.add(train_step)
wf.add(predict_step)
wf.submit()
"""
