#ifndef NET_H
#define NET_H

#include <string>
#include <random>

#include "Layers.h"

class Net{
	Loss *lossFunc = nullptr;
	Layer **layers = nullptr;
	int depth, current;
	double learnRate = 0.001;
	double regStrength = 0.00001;
	double learnRateDecay = 0.999;
	int batchSize = 200;
	int num_iter = 1000;

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
		learnRate *= learnRateDecay;
	}

	void train(Matrix &data, Vector &labels){
		Vector current;
		int correct = 0, guess;
		double loss = 0, score;
		for(int y=0; y<data.yDim(); ++y){
			current.copy(data[y]);
			// current = data[y];
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

	double validate(Matrix &valid, Vector &labels){
		int guess, correct = 0, index;
		std::random_device rd;
		std::mt19937 rand(rd());
		for(int i=0; i<1000; ++i){
			index = rand()%valid.yDim();
			guess = predict(valid[index]);
			if(guess == (int)labels[index]) ++correct;
		}
		return (double)correct/10;
	}

	void train(Matrix &data, Vector &labels, Matrix &valid, Vector &validLabels){
		Vector current;
		std::random_device rd;
		std::mt19937 rand(rd());
		double score, loss = 0, validPercent;
		int guess, correct = 0;
		for(int i=0, index; i<num_iter*batchSize; ++i){
			index = rand()%data.yDim();
			current.copy(data[index]);
			// current = data[index];
			lossFunc->setCorrect(labels[index]);
			
			for(int j=0; j<depth; ++j) layers[j]->forward(current);
			lossFunc->forward(current);
			
			score = current[0];
			guess = 0;
			for(int j=1; j<current.length(); ++j){
				if(current[j] > score){
					score = current[j];
					guess = j;
				}
			}
			
			if(guess == (int)labels[index]) ++correct;

			current.fill(0);
			
			lossFunc->backward(current);
			loss+=lossFunc->getLoss();
			for(int j=depth-1; j>=0; --j) layers[j]->backward(current);
			
			if((i+1)%batchSize == 0){
				update();
				validPercent = validate(valid, validLabels);
				std::cout << "Batch Loss: " << (double)loss/batchSize << 
								"\nTest Accuracy: " << (double)100.0*correct/batchSize <<
								"\nValidation Accuracy: " << validPercent <<
								"\nLearn Rate: " << learnRate << "\n\n";
				correct = 0;
				loss = 0;
			}
		}
	}

	~Net(){
		for(int i=0; i<depth; ++i) delete layers[i];
		delete[] layers;
		delete lossFunc;
	}
};

#endif