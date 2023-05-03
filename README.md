# Non-Intrusive Load Monitoring <br /> based on Residential Electricity Consumption
**A dissertation submitted to the University of Bristol in accordance with the BSc in Computer Science degree requirements. <br/>
Author: Nuno Rodrigues Alberto**

<br />

## Abstract

Approximately a quarter of the global electric energy consumption is thought to be consumed for residential purposes, and this figure has been rising in recent years. The usage of smart meters to provide real-time energy consumption feedback can cut energy use by up to 17% and it has been demonstrated that appliance-level insights result in even greater energy usage reductions.

Energy disaggregation, also referred to as non-intrusive load monitoring (NILM), is the process of inferring appliance-level energy consumption from the aggregated load drawn by a household. Deep learning has become increasingly popular for performing NILM due to outperforming traditional approaches and overcoming some of the setbacks inherent to them, such as the need for expert knowledge. Techniques like pattern matching and blind source separation often have difficulties in a variety of real-life scenarios due to factors such as measurement equipment noise and substantial variations in power signatures between appliances of the same type.

In this dissertation, a deep learning approach is proposed, with its main component being a deep neural network composed of convolutional, long short-term memory and dense layers as well as regularisation and normalisation blocks. The method also contained pre- and post-processing stages and it is trained and tested on data from four distinct appliances from the reference energy disaggregation data set (REDD), the most commonly used data set in NILM research. The proposed approach is shown to achieve stateof-the-art results, with performance metrics and training times being improved by as much as 41% and 88%, respectively.

<br />

## Main Achievements and Contributions

Some of the main achievements and contributions of this dissertation are listed below:

- Learnt how to apply deep learning to a real-life problem, namely energy disaggregation, as a third-year Bachelor’s student without a background in the field; worthy of note is that BSc students are not eligible to enrol in the deep learning unit.

- Engaged with NILM researchers from an external organisation and entered into productive dialogues with code and pre-processed data being shared.

- Remarkably improved the approach suggested by Zhang et al. [1], the baseline for seq2point energy disaggregation approaches.

- Obtained state-of-the-art results on the reference energy disaggregation data set, the most popular data set within the NILM research community.

<br />

## Supporting Technologies

The following supporting technologies were used under the scope of this dissertation:

- The Anaconda software distribution was utilised in order to facilitate the management and
deployment of Python packages and it can be accessed [here](https://www.anaconda.com/).

- The TensorFlow open-source machine learning library was used to implement the deep neural
network proposed in this thesis and the software is available from [here](https://www.tensorflow.org/).

- The non-intrusive load monitoring workflow suggested by Zhang et al. [1] was used as a starting
point for the proposed approach and its source code can be found [here](https://github.com/MingjunZhong/seq2point-nilm).

<br />

References:

[1] Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles Sutton. Sequence-to-Point Learning with Neural Networks for Non-Intrusive Load Monitoring. In Proceedings of the 32nd AAAI Conference on Artificial Intelligence, 2018. 

<br />

If any questions arise, please do not hesitate in reaching out to me via email: nunoo.albertoo@gmail.com or gz20955@bristol.ac.uk.
