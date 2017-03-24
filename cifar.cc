#include <iostream>
#include <fstream>
#include <string>

#include "Matrix.h"
#include "Layers.h"
#include "Net.h"

using namespace std;

string label[10];
Matrix train1(3072, 10000);
Vector labels1(10000);
Matrix train2(3072, 10000);
Vector labels2(10000);
Matrix train3(3072, 10000);
Vector labels3(10000);
Matrix train4(3072, 10000);
Vector labels4(10000);
Matrix train5(3072, 10000);
Vector labels5(10000);
Matrix test(3072, 10000);
Vector testLabels(10000);

void getLabels(){
	ifstream fin;
	fin.open("cifar-10/batches.meta.txt");
	if(!fin){
		cout << "ERROR: Label input" << endl;
		return;
	}
	for(int i=0; i<10 && !fin.eof(); i++) getline(fin, label[i]);
	fin.close();
}

void getData(Vector &labels, Matrix &data, string filename){
	ifstream fin;
	fin.open("cifar-10/" + filename);
	if(!fin){
		cout << "ERROR: Data input" << endl;
		return;
	}
	for(int i=0; !fin.eof(); i++){
		if(i%3073 == 0) labels[(int)(i/3073)] = fin.get();
		else data((int)(i/3073), i%3073-1) = (double)fin.get();
	}
	fin.close();
}

void testNet(Net &net){
	int correct = 0, guess;
	for(int i=0; i<10000; ++i){
		guess = net.predict(test[i]);
		if(guess == testLabels[i]) ++correct;
	}
	cout << "Statistics: " << (double)correct/100.0 << '\n';
}

int main(){
	getLabels();
	getData(labels1, train1, "data_batch_1.bin");
	getData(labels2, train2, "data_batch_2.bin");
	getData(labels3, train3, "data_batch_3.bin");
	getData(labels4, train4, "data_batch_4.bin");
	getData(labels5, train5, "data_batch_5.bin");
	getData(testLabels, test, "test_batch.bin");

	Net net(new SoftMax(10), 6);
	net.push(new FullyConn(3072, 200));
	net.push(new BatchNorm(200));
	net.push(new ReLU(200));
	net.push(new FullyConn(200, 10));
	net.push(new BatchNorm(10));
	net.push(new ReLU(10));

	for(int i=0; i<10; ++i){
		cout << i << '\n';
		net.train(train1, labels1);
		net.train(train2, labels2);
		net.train(train3, labels3);
		net.train(train4, labels4);
		net.train(train5, labels5);
	}
	testNet(net);
}