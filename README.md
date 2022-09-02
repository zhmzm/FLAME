# FLAME

Unofficial implementation for paper FLAME: Taming Backdoors in Federated Learning, if there is any problem, please let me know.

paper FLAME: Taming Backdoors in Federated Learning is from https://www.usenix.org/system/files/sec22-nguyen.pdf

Please contact me if you have any difficulty to run the code in issue.

## Requirement

Python=3.9

pytorch=1.10.1

scikit-learn=1.0.2

opencv-python=4.5.5.64

Scikit-Image=0.19.2

matplotlib=3.4.3

hdbscan=0.8.28

jupyterlab=3.3.2

Install instruction are recorded in install_requirements.sh

## Run

VGG and ResNet18 can only be trained on CIFAR-10 dataset, while CNN can only be trained on fashion-MNIST dataset.

```
python main_fed.py      --dataset cifar,fashion_mnist \
                        --model VGG,resnet,cnn \
                        --attack baseline,dba \
                        --lr 0.1 \
                        --malicious 0.1 \
                        --poison_frac 1.0 \
                        --local_ep 2 \
                        --local_bs 64 \
                        --attack_begin 0 \
                        --defence avg,fltrust,flame,krum,RLR \
                        --epochs 200 \
                        --attack_label 5 \
                        --attack_goal -1 \
                        --trigger 'square','pattern','watermark','apple' \
                        --triggerX 27 \
                        --triggerY 27 \
                        --gpu 0 \
                        --save save/your_experiments \
                        --iid 0,1 
```

Images with triggers on attack process and test process are shown in './save' when running.
Results files are saved in './save' by default, including a figure and a accuracy record.
More default parameters on different defense strategies or attack can be seen in './utils/options'.