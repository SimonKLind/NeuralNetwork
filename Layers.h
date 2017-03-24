#ifndef LAYERS_H
#define LAYERS_H

#include <random>
#include <cmath>
// #include <iostream>

#include "Matrix.h"

class Layer{

protected:
	Vector latestIn;
	int in, out;

public:

	Layer(int inDim, int outDim): in(inDim), out(outDim){
		latestIn.make(inDim);
	}

	virtual void forward(Vector &input) = 0;
	virtual void backward(Vector &grads) = 0;
	virtual void update(double learnRate, double regStrength){}
};

class FullyConn: public Layer{
	Matrix weights;
	Matrix gradients;
	Vector biases;
	Vector biasGrads;

public:

	FullyConn(int inDim, int outDim): Layer(inDim, outDim){
		std::random_device rd;
		std::mt19937 rand(rd());
		std::normal_distribution<double> dist(0.0, 1.0);
		biases.make(outDim);
		biases.fill(0);
		biasGrads.make(outDim);
		biasGrads.fill(0);
		// std::cout << "Hej\n";
		weights.make(inDim, outDim);
		gradients.make(inDim, outDim);
		double rootN = sqrt((double)2/inDim);
		for(int y=0; y<outDim; ++y){
			for(int x=0; x<inDim; ++x){
				// weights(y, x) = (int)rand()%inDim*sqrt((double)2/inDim);
				weights(y, x) = dist(rand)*rootN;
				gradients(y, x) = 0;
			}
		}
	}

	void forward(Vector &input){
		this->latestIn = input;
		weights.dot(input);
		input+=biases;
	}

	void backward(Vector &grads){
		Vector global(grads);
		grads.resize(this->in);
		for(int y=0; y<this->out; ++y){
			for(int x=0; x<this->in; ++x){
				gradients(y, x) += this->latestIn[x]*global[y];
				grads[x] += weights(y, x)*global[y];
			}
			biasGrads[y] += global[y];
		}
	}

	void update(double learnRate, double regStrength){
		for(int y=0; y<this->out; ++y){
			for(int x=0; x<this->in; ++x){
				// std::cout << weights(y, x) << " -> ";
				weights(y, x) -= gradients(y, x)*learnRate + weights(y, x)*regStrength;
				// std::cout << weights(y, x) << " : " << gradients(y, x) << '\n';
				gradients(y, x) = 0;
			}
			biases[y] -= biasGrads[y]*learnRate/* + biases[y]*regStrength*/;
		}
	}
};

class BatchNorm: public Layer{
	double variance;
	double ivar;
	double mean;

	double y, b;
	double dy, db;
	Vector out;

	double sumGlobal, sumGxOut;

	void calcMean(Vector &batch){
		mean = 0;
		for(int i=0; i<this->in; ++i) mean += batch[i];
		mean /= this->in;
	}

	void calcVar(Vector &batch){
		variance = 0;
		for(int i=0; i<this->in; ++i) variance += (batch[i]-mean)*(batch[i]-mean);
		variance /= this->in;
	}

public:

	BatchNorm(int inDim): Layer(inDim, inDim){
		y=1;
		b=dy=db=0;
		out.make(inDim);
	}

	void forward(Vector &input){
		calcMean(input);
		calcVar(input);
		ivar = 1/sqrt(variance);
		for(int i=0; i<this->in; ++i){
			this->latestIn[i] = input[i];
			out[i] = (input[i]-mean)*ivar;
			input[i] = out[i]*y+b;
		}
	}

	void backward(Vector &grads){
		Vector global(grads);
		sumGlobal = 0;
		sumGxOut = 0;
		for(int i=0; i<this->in; ++i){
			sumGlobal+=global[i];
			sumGxOut+=global[i]*out[i];
		}
		double com = y*ivar/this->in;
		for(int i=0; i<this->in; ++i){
			dy+=out[i]*global[i];
			db+=global[i];
			grads[i] = com*(global[i]*this->in-sumGlobal-out[i]*sumGxOut);
		}
	}

	void update(double learnRate, double regStrength){
		y -= dy*learnRate+y*regStrength;
		b -= db*learnRate/*+b*regStrength*/;
		dy = db = 0;
	}
};

class ReLU: public Layer{
	
public:

	ReLU(int inDim): Layer(inDim, inDim) {}

	void forward(Vector &input){
		for(int i=0; i<this->in; ++i){
			if(input[i] < 0) input[i] = 0;
			this->latestIn[i] = (input[i] > 0) ? 1 : 0;
		}
	}

	void backward(Vector &grads){
		grads *= this->latestIn;
	}
};

class Loss: public Layer{

protected:
	double loss;
	int correct = -1;

public:

	Loss(int inDim): Layer(inDim, inDim){}

	virtual void forward(Vector &input) = 0;
	virtual void backward(Vector &grads) = 0;

	double getLoss(){
		return loss;
	}

	double setCorrect(int index){
		correct = index;
	}
};

class SVM: public Loss{

public:

	SVM(int inDim): Loss(inDim){}

	void forward(Vector &input){
		this->latestIn = input;
	}

	void backward(Vector &grads){
		loss = 0;
		double current;
		for(int i=0; i<this->in; ++i){
			if(i != correct){
				current = this->latestIn[i]-this->latestIn[this->correct]+1;
				if(current > 0){
					grads[i] = 1;
					grads[this->correct] -= 1;
					this->loss+=current;
				}
			}
		}
	}
};

class SoftMax: public Loss{
	double total;

public:

	SoftMax(int inDim): Loss(inDim){}

	void forward(Vector &input){
		double max = input[0];
		for(int i=1; i<this->in; ++i){
			if(input[i] > max) max = input[i];
		}
		total = 0;
		for(int i=0; i<this->in; ++i){
			this->latestIn[i] = exp(input[i]-max);
			total+=this->latestIn[i];
		}
		// loss = -log(this->latestIn[correct]/total);
	}

	void backward(Vector &grads){
		this->loss = -log(this->latestIn[this->correct]/total);
		for(int i=0; i<this->in; ++i) grads[i] = this->latestIn[i]/total;
		grads[this->correct] -= 1;
	}
};

#endif