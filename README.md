# JudgeAnything
See our demo at [web](https://urrealhero.github.io/judgeanythingweb/)

## Todo
- [x] ___Release the Benchmark and the Dataset___
- [x] ___Gather code for auto-evaluation on our benchmark___
- [ ] ___Release the OmniArena platform and general pair comparison code___

## Data
We have released our benchmark and models responses in 
[dataset](https://huggingface.co/datasets/pudashi/JudgeAnything), you should download them into `dataset/`


## Reproduction of auto-evaluation
You can use scripts under `scripts/` to re-run the evaluation on existing data. 

### API
We now provide the reproduction code for `Gemini-1.5-pro`, `Gemini-2.0-flash`, `Gemini-2.0-flash-lite`.

### Local 
We also provides an extensive huggingface-support interface to evaluate, the prompt template is `Phi4v-multimodal`, you can modify the local special token template in `scripts/utils/config.py` or modify the procedure of building local prompt template in `scripts/utils/prompt_builder.py`.

## General Evaluation
To evaluate your own private generated data, transform the data into following format to do the score evaluation.(We are still working on gathering general code for our OmniArena's pair comparison code). We encourage you to get the prompt in `scripts/utils/prompt_builder.py` and evaluate yourself
```json
[
    {
        "uniq_id": "the uniq_id of task",
        "response_id": "create a response_id for your response",
        "task_name": "task_name of task",
        "model_name": "your model name",
        "response": {
            "type": "text/image/video/audio",
            "content": "string or list, if non-text type, provide the abspath here"
        }
    }
]
```
