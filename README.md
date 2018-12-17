# Bayesian CNN with Variational Inference and it's application

In this repo **Bayesian Convolutional Neural Network (BayesCNN) using Variational Inference** is proposed, that introduces probability distribution over the weights. Furthermore, the proposed BayesCNN architecture is applied to tasks like **Image Classification, Image Super-Resolution and Generative Adversarial Networks**.

BayesCNN is based on **Bayes by Backprop** which derives a variational approximation to the true posterior. 
Our proposed method not only achieves performances equivalent to frequentist inference in identical architectures but also incorporate a measurement for uncertainties and regularisation. It further eliminates the use of dropout in the model. Moreover, we predict how certain the model prediction is based on the epistemic and aleatoric uncertainties and finally, we propose ways to prune the Bayesian architecture and to make it more computational and time effective. 

---------------------------------------------------------------------------------------------------------

## Folder Structure and Content

### Image Recognition

*   The Bayesian CNN is applied to the task of Image Recognition and the results are compared to frequentist architectures for MNIST, CIFAR10 and CIFAR100 datasets. 

*   A measure of uncertainty is added with the prediction and the epistemic and aleatoric uncertainty is estimated.

*   Bayesian AlexNet, LeNet and 3Conv3FC is proposed and applied to Image recognition tasks. 

*   Code and implementation details available at: [Bayesian CNN Image Recognition](https://github.com/kumar-shridhar/PyTorch-BayesianCNN/tree/master/Image%20Recognition)

---------------------------------------------------------------------------------------------------------

### Super Resolution

*   Bayesian CNN is applied to the task of Super Resolution on BSD300 dataset and the results are compared to other methods.

*   Implementation and code is available here : [PyTorch Bayesian Super Resolution](https://github.com/kumar-shridhar/PyTorch-BayesianCNN/tree/master/Super%20Resolution) 

---------------------------------------------------------------------------------------------------------

### Paper

*   Paper contains the paper about the Bayesian CNN with Variational Inference. The paper is also available on Arxiv: [Bayeisan CNN with Variational Inference](https://arxiv.org/abs/1806.05978)

*   Feel free to cite the author, if the work is any help to you:

```
@article{shridhar2018bayesian,
  title={Bayesian Convolutional Neural Networks with Variational Inference},
  author={Shridhar, Kumar and Laumann, Felix and Llopart Maurin, Adrian and Olsen, Martin and Liwicki, Marcus},
  journal={arXiv preprint arXiv:1806.05978},
  year={2018}
}
```

---------------------------------------------------------------------------------------------------------

### Thesis

* Thesis contains the detailed explaination of all the concepts mentioned from background knowledge to empirical analysis and conclusion. 

* Thesis chapters overview is available here: [Master Thesis BayesianCNN](https://github.com/kumar-shridhar/Master-Thesis-BayesianCNN)

---------------------------------------------------------------------------------------------------------

