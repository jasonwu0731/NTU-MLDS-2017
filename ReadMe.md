
MLDS Online Course
http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS17.html 

#Basic Structures for Deep Learning Models
LSTM/GRU
Target Delay: Only for unidirectional RNN 
LSTM > RNN > feedforward
Bi-direction > uni-direction
Training LSTM is faster than RNN
LSTM 比較不會有gradient vanish的問題
Forget gate is critical for performance
Output gate activation function is critical
Stack RNN: push/pop/nothing
Convolution Layer: Filter/ Stride
Pooling Layer: min/max/L2

#Computational Graph and Back-propagation
#Language Modeling: 估計一個word sequence的機率
N-gram LM: P(w1,w2,…) = P(W1|START)*P(W2|W2)*…
Challenge of N-gram: large model, not sufficient data
NN LM: 相較於N-gram，減少參數

#Special Deep Learning Structure
Spatial Transformer: 
1. CNN is not invariant to scaling and rotation
2. 訓練一個能產生6個參數的NN，來增加辨識度
Highway Network & Grid LSTM
1. Highway Network 就是把RNN竪直 (Residual Network)
2. Grid LSTM Memory for both time and depth
Recursive Network: EX: sentiment analysis

#Conditional Generation by RNN & Attention 
1. Attention: Dynamic Conditional Generation
利用機器學到的weight來針對部分input作加權
2. Memory Network:  Hopping 
3. Neural Turing Machine: Not only read from memory but also modify the memory through attention
Good Attention: each input component has approximately the same attention weight, ex: regularization 
4. Exposure Bias: mismatch between train and test (train時用正確答案當下個時間點的input，test時用現在的output當成下個時間點的input)
5. Scheduled Sampling: 隨機拿正確答案或model output來當下個時間點input
6. Beam Search: keep several best path at each step
7. Object level v.s. Component level

#Generative Adversarial Network (GANs)
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

b. Improved version




 