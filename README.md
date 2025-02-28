# Wolf 🐺: Dense Captioning with a World Summarization Framework
The repo contains official Implementation and dataset (including annotations) of [Wolf 🐺: Dense Captioning with a World Summarization Framework](https://link.springer.com/chapter/10.1007/978-3-031-25063-7_1).

## Wolf Benchmark Overview
| Dataset | Number | Download | Annotations |
|----------|----------|----------|----------|
| Normal Driving Scenes   | 4785   | [wolf_driving_index_file.json](https://github.com/NVlabs/Wolf/blob/main/annotations/wolf_driving_index_file.json)   | [Normal Nuscenes_annotations.json](https://github.com/NVlabs/Wolf/blob/main/annotations/Normal%20Nuscenes_annotations.json)   |
| Challenging Driving Scenes   | 500   | [wolf_driving_index_file.json](https://github.com/NVlabs/Wolf/blob/main/annotations/wolf_driving_index_file.json)   | [Interactive Nuscenes_annotations.json](https://github.com/NVlabs/Wolf/blob/main/annotations/Interactive%20Nuscenes_annotations.json)   |
| General Daily Scenes   | 473   | [link]()   | [Pexels_annotations.json](https://github.com/NVlabs/Wolf/blob/main/annotations/Pexels_annotations.json)   |
| Robot Manipulation   | 100   | [link]()   | [robotics_annotations.json](https://github.com/NVlabs/Wolf/blob/main/annotations/robotics_annotations.json)   |

Note: please download the driving dataset from [official Nuscenes webpage](https://www.nuscenes.org/nuscenes) based our provided index file [wolf_driving_index_file.json](https://github.com/NVlabs/Wolf/blob/main/annotations/wolf_driving_index_file.json).

## Capscore: Wolf metric to evaluate your caption quality
CapScore is a quantitative metric to use LLMs (GPT-4) to evaluate the similarity between predicted and human-annotated (ground-truth) captions. Please revise the script and run the code below to obtain your results.

`python get_capscore.py`

## Running 
Please revise the script and run the code below to obtain your results.

`python get_wolfcaps.py`


If you find this repo useful, please cite:
```
@article{li2024wolf,
  title={Wolf: Dense Captioning with a world summarization framework},
  author={Li, Boyi and Zhu, Ligeng and Tian, Ran and Tan, Shuhan and Chen, Yuxiao and Lu, Yao and Cui, Yin and Veer, Sushant and Ehrlich, Max and Philion, Jonah and others},
  journal={arXiv preprint arXiv:2407.18908},
  year={2024}
}
```
