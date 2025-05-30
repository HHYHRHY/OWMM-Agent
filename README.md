# OWMM-Agent

This repo maintains an overview of the OWMM-Agent project, as introduced in paper "OWMM-Agent: Open World Mobile Manipulation With Multi-modal Agentic Data Synthesis".

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
  - [Install Habitat environment and datasets](#install-habitat-environment-and-datasets)
  - [Install VLM dependencies](#install-vlm-dependencies)
- [Usage](#usage)
- [Credit](#credit)

## Introduction

The rapid progress of navigation, manipulation, and vision models has made mobile manipulators capable in many specialized tasks. 
However, the open-world mobile manipulation (OWMM) task remains a challenge due to the need for generalization to open-ended instructions and environments, as well as the systematic complexity to integrate high-level decision making with low-level robot control based on both global scene understanding and current agent state.
To address this complexity, we propose a novel multi-modal agent architecture that maintains multi-view scene frames and agent states for decision-making and controls the robot by function calling.
A second challenge is the hallucination from domain shift. To enhance the agent performance, we further introduce an agentic data synthesis pipeline for the OWMM task to adapt the VLM model to our task domain with instruction fine-tuning.
We highlight our fine-tuned OWMM-VLM as the first dedicated foundation model for mobile manipulators with global scene understanding, robot state tracking, and multi-modal action generation in a unified model. 
Through experiments, we demonstrate that our model achieves SOTA performance compared to other foundation models including GPT-4o and strong zero-shot generalization in real world.
In this repository, we provide the complete pipeline code for data collection and data annotation, as well as the code for step evaluation and simulator evaluation.

## Installation

You should first clone our repo:

```bash
git clone https://github.com/owmm-vlm-project/OWMM-Agent.git
```

### Install Habitat environment and datasets

Please follow the instructions in the [Install Habitat Environment](./install_habitat.md) to install the Habitat environment. Please refer to the Meta official repository [habitat-lab](https://github.com/facebookresearch/habitat-lab) for troubleshooting and more information. 

For extra dependencies in Habitat and original datasets used in OWMM-VLM, please follow the instructions in [Habitat-MAS Package](./dataset_generation/habitat-lab/habitat-mas/README.md).

### Install VLM dependencies

For the dependencies required for model fine-tuning and deployment, please refer to [InternVL2.5](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/). For the dependencies of the baselines, please refer to the dependency downloads of [PIVOT](https://huggingface.co/spaces/pivot-prompt/pivot-prompt-demo/tree/main), [Robopoint](https://github.com/wentaoyuan/RoboPoint), and [GPT](https://platform.openai.com/docs/quickstart).

## Usage

For dataset generation and simulator evaluation, Please follow the instructions in [sim](./sim/README.md). After sampling dataset from dataset generation, please refer to the instructions in [dataset_annotation](./dataset_annotation/README.md) to obtain annotated datasets. For step evaluation, please follow the instructions in [step_evaluation](./dataset_evalutation/README.md).

## Credit

This repo is built upon [EMOS](https://github.com/SgtVincent/EMOS), which built based on the [Habitat Project](https://aihabitat.org/) and [Habitat Lab](https://github.com/facebookresearch/habitat-lab) by Meta. 
We would like to thank the authors of EMOS and the original Habitat project for their contributions to the community.
