#### This repository contains some baseline methods for audio-visual event localization and video parsing tasks. It implements the [AVEL](https://github.com/YapengTian/AVE-ECCV18), [AVSDN](https://arxiv.org/pdf/1902.07473.pdf), [CMRAN](https://github.com/FloretCat/CMRAN), [PSP](https://github.com/jasongief/PSP_CVPR_2021), CPSP, and SSPSP methods.



-----
### for Audio-Visual Event Localization 
#### fully supervised setting
- Train
```
cd cpsp_avel
bash run_fully_supv.sh
```

#### weakly supervised setting
- Train
```
cd cpsp_avel
bash run_weakly_supv.sh
```

------
### for Audio-Visual Video Parsing
- Train
```
cd cpsp_avvp
bash run.sh
```

We also provide the pretrained psp/cpsp models for these tasks, you can download it from [here](https://drive.google.com/drive/folders/1sMMild9eZ6WEj_9b5QW1ZMYNWzw0Ah54?usp=sharing).


------
### Citation
Please consider cite following paper if it is helpful in your research or projects:
```
@inproceedings{zhou2021positive,
  title={Positive Sample Propagation along the Audio-Visual Event Line},
  author={Zhou, Jinxing and Zheng, Liang and Zhong, Yiran and Hao, Shijie and Wang, Meng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8436--8444},
  year={2021}
}
```
