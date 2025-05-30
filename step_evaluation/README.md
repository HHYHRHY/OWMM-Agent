# Step Evaluation

The repository contains folders with inference and evaluation methods, including our OWMM-VLM models,PIVOT agent and Robopoint agent. 

You can generate testset in simulator follow the instruction in [sim README](../sim/README.md).We privode the testset from real world in [step_evaluation_real.zip](https://drive.google.com/drive/folders/1FIvCu28nRcz_tSsMLbii3pigFwLtwUQU?usp=sharing).

You can execute these scripts to test models in single step evaluation settings. You are required to deploy models and associate it with the appropriate URL in `xxx_inference.py`, and add your API_KEY and BASE_URL to call gpt-4o.
