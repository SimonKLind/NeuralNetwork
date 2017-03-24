#ifndef NET_H
#define NET_H

#include <string>

#include "Layers.h"

class Net{
	Loss *lossFunc = nullptr;
	Layer **layers = nullptr;
	int depth, current;
	double learnRate = 0.001;
	double regStrength = 0.00001;

public:

	Net(Loss *lossFunc, int numLayers): depth(numLayers), current(0){
		this->lossFunc = lossFunc;
		layers = new Layer*[numLayers];
	}

	void push(Layer *layer){
		// if(current >= depth) return;
		layers[current++] = layer;
	}

	int predict(Vector &data){
		Vector temp(data);
		for(int i=0; i<depth; ++i) layers[i]->forward(temp);
		double score = temp[0];
		int index = 0; 
		for(int i=1; i<temp.length(); ++i){
			if(temp[i] > score){
				index = i;
				score = temp[i];
			}
		}
		return index;
	}

	void update(){
		for(int i=0; i<depth; ++i) layers[i]->update(learnRate, regStrength);
		learnRate *= 0.999999;
	}

	void train(Matrix &data, Vector &labels){
		Vector current;
		int correct = 0, guess;
		double loss = 0, score;
		for(int y=0; y<data.yDim(); ++y){
			current.copy(data[y]);
			lossFunc->setCorrect(labels[y]);

			for(int i=0; i<depth; ++i) layers[i]->forward(current);
			lossFunc->forward(current);

			score = current[0];
			guess = 0;
			for(int i=1; i<current.length(); ++i){
				if(current[i] > score){
					score = current[i];
					guess = i;
				}
			}

			if(guess == labels[y]) ++correct;

			current.fill(0);

			lossFunc->backward(current);
			loss += lossFunc->getLoss();
			for(int i=depth-1; i>=0; --i) layers[i]->backward(current);

			if((y+1)%10 == 0) update();
		}
		std::cout << "Loss: " << loss << " Stats: " << (double)correct/100 << '\n';
	}

	~Net(){
		for(int i=0; i<depth; ++i) delete layers[i];
		delete[] layers;
		delete lossFunc;
	}
};

#endif