#include <iostream>
#include <fstream>
#include <string>

#include "Layers.h"
#include "Net.h"
#include "Matrix.h"

using namespace std;

Matrix train(3072, 50000);
Vector trainLabels(50000);
Matrix valid(3072, 10000);
Vector validLabels(10000);

void getData(){
	ifstream fin;
	string filename;
	for(int i=0; i<5; ++i){
		filename = "cifar-10/data_batch_";
		filename += to_string(i+1);
		filename += ".bin";
		fin.open(filename);
		if(!fin){
			cout << "Error loading " << i << '\n';
			return;
		}
		// cout << "loading " << i << '\n';
		for(int j=0; !fin.eof(); ++j){
			if(j%3073 == 0) trainLabels[(int)(j/3073)+10000*i] = fin.get();
			else train((int)(j/3073)+10000*i, j%3073-1) = (double)fin.get();
		}
		fin.close();
	}
	fin.open("cifar-10/test_batch.bin");
	if(!fin){
		cout << "Error loading validation\n";
		return;
	}
	for(int i=0; !fin.eof(); ++i){
		if(i%3073 == 0) validLabels[(int)(i/3073)] = fin.get();
		else valid((int)(i/3073), i%3073-1) = (double)fin.get();
	}
	fin.close();
}

void testNet(Net &net){
	int correct = 0, guess;
	for(int i=0; i<10000; ++i){
		guess = net.predict(valid[i]);
		if(guess == (int)validLabels[i]) ++correct;
	}
	cout << "Statistics: " << (double)correct/100.0 << '\n';
}

void normalize(){
	Vector means(3072);
	means.fill(0);
	for(int y=0; y<50000; ++y){
		for(int x=0; x<3072; ++x) means[x] += train(y, x);
	}
	for(int i=0; i<3072; ++i) means[i] /= 50000;
	for(int i=0; i<50000; ++i){
		if(i<10000) valid[i] -= means;
		train[i] -= means;
	}
}

int main(){
	getData();
	normalize();
	Net net(new SoftMax(10), 6);
	net.push(new FullyConn(3072, 200));
	net.push(new BatchNorm(200));
	net.push(new ReLU(200));
	net.push(new FullyConn(200, 10));
	net.push(new BatchNorm(10));
	net.push(new ReLU(10));
	net.train(train, trainLabels, valid, validLabels);
	// net.train(train, trainLabels);
	testNet(net);
}