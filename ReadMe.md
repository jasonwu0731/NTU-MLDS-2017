
MLDS Online Course
==================
http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS17.html 

## Basic Structures for Deep Learning Models
> LSTM/GRU
> Target Delay: Only for unidirectional RNN 
> LSTM > RNN > feedforward
> Bi-direction > uni-direction
> Training LSTM is faster than RNN
> LSTM 比較不會有gradient vanish的問題
> Forget gate is critical for performance
> Output gate activation function is critical
> Stack RNN: push/pop/nothing
> Convolution Layer: Filter/ Stride
> Pooling Layer: min/max/L2

## Computational Graph and Back-propagation

## Language Modeling: 估計一個word sequence的機率
> N-gram LM: P(w1,w2,…) = P(W1|START)*P(W2|W2)*…
> Challenge of N-gram: large model, not sufficient data
> NN LM: 相較於N-gram，減少參數

## Special Deep Learning Structure
### Spatial Transformer: 
1. CNN is not invariant to scaling and rotation
2. 訓練一個能產生6個參數的NN，來增加辨識度
### Highway Network & Grid LSTM
1. Highway Network 就是把RNN竪直 (Residual Network)
2. Grid LSTM Memory for both time and depth
### Recursive Network: EX: sentiment analysis

## Conditional Generation by RNN & Attention 
1. Attention: Dynamic Conditional Generation
利用機器學到的weight來針對部分input作加權
2. Memory Network:  Hopping 
3. Neural Turing Machine: Not only read from memory but also modify the memory through attention
Good Attention: each input component has approximately the same attention weight, ex: regularization 
4. Exposure Bias: mismatch between train and test (train時用正確答案當下個時間點的input，test時用現在的output當成下個時間點的input)
5. Scheduled Sampling: 隨機拿正確答案或model output來當下個時間點input
6. Beam Search: keep several best path at each step
7. Object level v.s. Component level

## Generative Adversarial Network (GANs)
1. Basic Idea of GAN:
a generator G is a network. The network defines a probability distribution.
the loss of discriminator is related to JS divergence.
2. Algorithm of GAN
In each training iteration
a. (Learning D, may repeat k times)
	Sample m examples from data distribution, x1,x2,…xm
	Sample m noise samples from some distribution, z1,z2,…,zm
	Obtain G data where x~i = G(zi)
	Update D to maximize V = 1/m( SUM(logD(xi)) + SUM(log(1-D(x~i))) )
b. (Learning G, only once)
	Sample another m noise samples, z1,z2,…,zm
	Update G to minimize V = 1/m( SUM(log(1-D(G(zi))) ) )
3. Unified Framework
a. f-divergence: 量測兩個distribution有多不一樣，越大越不一樣 
	D_f(P||Q) = Integ_x (q(x)f(p(x)/q(x))) dx, f is convex(凸), f(1)=0
	KL divergence: f=xlogx
	Reverse KL: f=-logx
	Chi Square: f=(x-1)^2  
b. Frenchei Conjugate: f*(t) = max{xt-f(x)} for x belongs to dom(f)
	every convex function f has a conjugate function f*
	if f(x)=xlogx, f*(t)=exp(t-1)
	(f*)* = f
c. Connect to GAN
	D_f(P||Q) = Integ_x (q(x)f(p(x)/q(x))) dx 
	          = Integ_x (q(x)(max{p(x)*t/q(x)-f*(t)})) dx 
			for t belongs to dom(f*)
	          ~ max (Integ_x(p(x)D(x))dx - Integ_x(q(x)f*(D(x)))dx ) 
			for any D(x)
	Original GAN 只是 f-divergence GAN的特例
	Double loop v.s. Single-Step 
	用不同的Divergence可以得到不同結果的分佈
4. WGAN: 
Using Earth Mover’s Distance(Wasserstein Distance) to evaluate two distribution
a. Original version
	W(P_data, P_G)=max{E_x~P_data[D(x)]-E_x~P_g[D(x)]} 
		for D belongs to 1-Lipschitz
	Lipschitz Function: |f(x1)-f(x2)|<=K|x1-x2|, output v.s. input 
	How to use gradient descent to optimize W(P_data, P_G) ?
	I. Weight Clipping: force wrights w between c and -c
		缺點是無法真正找到function D讓函數最大化，但沒clipping會無法讓training停止
	Algorithm of WGAN: Almost same as GAN, except
	I. Update D to maximize V = 1/m( SUM(D(xi)) - SUM((D(x~i))) ) 
	II. No sigmoid for the output of D
	III. Weight clipping while updating
	IV. Update G to minimize V = 1/m( SUM(-D(G(zi))) ) 
	WGAN 可以衡量Discriminator的loss來決定train的成果
b. Improved version
	A 1-Lipschitz function is differentiable iff it has gradients with norm less than or equal to 1 everywhere
	D belongs to 1-Lipschitz <=> |grad_x D(x)| <= 1 for all x
	加上一個 penalty來制衡上式，有點像regularization
	Only give gradient constraint to the region between Pdata and PG because they influence how PG moves to Pdata
	W(P_data, P_G)=max{E_x~P_data[D(x)]-E_x~P_g[D(x)] - lambda*E_x~P_penalty[(|grad_x D(x)|-1)^2]} for any D
	也就是讓他越趨近1越好
c. 用GAN產生Sentence
把word or char做embedding然後看成圖片
產生唐詩蠻有趣的
其他方法：seqGAN，但比較像是reinforcement learning
d. Conditional GAN （pair data）
	ex1: txt to omg, input “train” and distribution z together
	兩種方法train D, 可以吃單一或多個input，多個就是同時要滿足txt跟img，多個應該會比較好
	ex2: img to img translation 	
e. Unpaired Transformation
	ex: Cycle GAN (Disco GAN)，同時讓G學習另一個domain的東西，也同時要能夠重新decode回原本的domain
	可以讓馬變斑馬，斑馬變馬
5. GAN and Feature Representation
a. GAN+Autoencoder
	"Generative Visual Manipulation on the Natural Image Manifold", ECCV, 2016.
	VAE+GAN: “Autoencoding beyond pixels using a learned similarity metric”, ICML. 2016
b. InfoGAN
	G的input是info+某個分部，output另外多接一個encoder希望它輸出原本的info
	可以找到控制GAN生成的東西的特定參數info parameter
c. BiGAN
	分別train一個encoder跟decoder，然後讓D分辨哪個x,z是真的input
6.Energy-based GAN (EBGAN)
	Viewing the D as an energy function (negative evaluation function) 
	用auto encoder當D，update D to minimize L_D(x,z)=D(x)+max(0,m-D(G(z)))
	Pulling-away term for training G
	Margin Adaptation GAN (MAGAN): 浮動的margin m
	Loss-sensitive GAN
7. Ensemble of GAN
	訓練多個GAN互相平行的訓練
	其他： Multi-Agent Diverse GAN, Message Passing Multi-Agent GAN, AdaGAN, Unrolled GAN
8. RL and GAN for Sentence Generation and Chat-bot
	“Deep Reinforcement Learning for Dialogue Generation“, EMNLP 2016
a. Policy Gradient/ Add a baseline:
	Objective function: 1/N * SUM{R(h_i,x_i) logP_theta(x_i|h_i) }
b. SeqGAN
	“SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient”, AAAI, 2017
	 “Adversarial Learning for Neural Dialogue Generation”, arXiv preprint, 2017
	Reward for every generation step: Monte Carlo Search / Discriminator for Partially Decoded Sequences
	Teacher Forcing
	Synthetic data
c. Original GAN






 