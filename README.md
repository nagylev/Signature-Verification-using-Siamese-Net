# Signature Verification using Siamese Net on SVC2004

This is a university project for the subject [**Training Project Laboratory (VITMAV45)**](https://portal.vik.bme.hu/kepzes/targyak/VITMAV45/). Our goal was to recreate and implement the publication of [**Dey, Dutta, Toledo, Ghosh, Lladós, Pal (2017)**](https://arxiv.org/pdf/1707.02131.pdf). In the paper they use Convolutional Siamese Network for Writer Independent Offline Signature Verification for offline datasets. We implemented their method to a dataset, which were generated from an online signature dataset **(SVC2004)**, by plotting the x and y coordinates. The results were less accurate than expected.

## Authors
 - [**Nagy Levente (ED1RB4)**](https://github.com/nagylev)
 - [**Márton Tárnok (GGDVB2)**](https://github.com/tamarci)

 ### Documentation
 [Hungarian Documentation] of the project.
 
 ### Generated dataset from SVC2004
  - We have 40 signers
  - Each with 20 genuine and 20 forged signatures
  - The generator we used ([**SigStatGUI**]()) generated the plot of the signatures, showing also pressure and interpolation
  - The program generates pictures with different size
  
    Genuine Signature                                                                                                                                  |  Forged Signature                                                                                                                                  :-----------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------:
<img  alt="image" src="https://user-images.githubusercontent.com/56648499/146528230-c353877e-2b45-4978-99ef-06623fa2eaa8.png" width="150" height="150">|<img alt= "image" src="https://user-images.githubusercontent.com/56648499/146528312-8abc6a5a-897e-4c54-91a4-4d8ef417bd20.png" width="150" height="150">
 


