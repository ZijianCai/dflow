

from dflow import ShellOPTemplate, InputParameter, InputArtifact, OutputParameter, OutputArtifact

"""
train = ShellOPTemplate(
    name="train",
    image="kianpu/cgcnn:latest",
    script="cd predict_artifact && . ~/.bashrc && python predict.py \
            formation-energy-per-atom.pth.tar \
            sample-regression \
            "
)
"""

train = ShellOPTemplate(
    name="train",
    image="kianpu/cgcnn:latest",
    script=". ~/.bashrc && python /train_artifact/main.py --train-size 10 --val-size 1 --test-size 1 \
            /train_artifact/root_dir && ls"
    # Will generate /model_best.pth.tar; checkpoint.pth.tar; test_results.csv
    # /predict_artifact/predict.py \
            #/predict_artifact/model_best.pth.tar \
            #/predict_artifact/data_to_be_predicted/sample-regression \
            
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
#predict_artifact = upload_artifact(["predict.py","data_to_be_predicted","model_best.pth.tar","cgcnn"])

train_artifact = upload_artifact(["cgcnn", "root_dir", "main.py"])
predict_artifact = upload_artifact(["predict.py","data_to_be_predicted","cgcnn"])
#simple_example_templ.inputs.parameters = {"msg": InputParameter()}
predict.inputs.artifacts = {"predict_inp_art": InputArtifact(path="predict_artifact"), \
                            "predict_model": InputArtifact(path="/model_best.pth.tar")}
train.inputs.artifacts = {"train_inp_art": InputArtifact(path="train_artifact")}
# define output
#simple_example_templ.outputs.parameters = {
#    "msg": OutputParameter(value_from_path="/tmp/msg.txt")
#}
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

