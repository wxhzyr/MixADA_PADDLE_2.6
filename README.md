# MixADA_PADDLE_2.6
This is a repository of MixADA in PADDLE2.6 version

# Better Robustness by More Coverage: Adversarial Training with Mixup Augmentation for Robust Fine-tuning

This is the repo for reproducing the results in our paper:
Better Robustness by More Coverage: Adversarial Training with Mixup Augmentation for Robust Fine-tuning ([arxiv](https://arxiv.org/abs/2012.15699)). ACL 2021 (Findings).

## Dependencies 

I conducted all experiments under Paddle-gup==2.6.2.

## Data

We provide the exact data that we used in our experiments for easier reproduction. The download link is [here](https://drive.google.com/file/d/1MIFljjU8sOzxZshBvq7gFqX9MidqUSFe/view?usp=sharing).

## Running 

I have included examples of how to run model training with MixADA as well as how to evaluate the models under adversarial attacks in `run_job.sh`. However, you need to modify the scripts to fill in your dataset and pretrained model checkpoint paths.

## Reference

Please consider citing our work if you found this code or our paper beneficial to your research.

```
@inproceedings{Si2020BetterRB,
  title={Better Robustness by More Coverage: Adversarial Training with Mixup Augmentation for Robust Fine-tuning},
  author={Chenglei Si and Zhengyan Zhang and Fanchao Qi and Zhiyuan Liu and Yasheng Wang and Qun Liu and Maosong Sun},
  booktitle={Findings of ACL},
  year={2021},
}
```

## Contact

If you encounter any problems, feel free to raise them in issues or contact the authors.
