#ifndef MATRIX_H
#define MATRIX_H

// #include <iostream>

class Vector{
	double *vec = nullptr;
	int size;

public:

	Vector(){}

	Vector(int length): size(length){
		vec = new double[length];
		// std::cout << "Done making: " << &vec << ", " << length << '\n';
	}

	Vector(Vector &other){
		// std::cout << "In copy constructor\n";
		size = other.size;
		vec = new double[size];
		// std::cout << "Done making: " << &vec << '\n';
		for(int i=0; i<size; ++i) vec[i] = other.vec[i];
	}

	void operator=(Vector &other){
		// std::cout << "In equals-operator\n";
		for(int i=0; i<size; ++i) vec[i] = other.vec[i];
	}

	void make(int length){
		// std::cout << "Deleting: " << &vec << '\n';
		delete[] vec;
		size = length;
		vec = new double[size];
		// std::cout << "Done making: " << &vec << ", " << size << '\n';
	}

	void copy(Vector &other){
		delete[] vec;
		size = other.size;
		vec = new double[size];
		for(int i=0; i<size; ++i) vec[i] = other.vec[i];
	}

	void resize(int newSize){
		// std::cout << "Deleting: " << &vec << '\n';
		delete[] vec;
		size = newSize;
		vec = new double[size];
		// std::cout << "Done making: " << &vec << '\n';
		fill(0);
	}

	void fill(double val){
		for(int i=0; i<size; ++i) vec[i] = val;
	}

	double dot(const Vector &other){
		// if(other.size != size) return 0;
		double res = 0;
		for(int i=0; i<size; ++i) res+=vec[i]*other.vec[i];
		return res;
	}

	double& operator[](int index){
		// if(index >= size || index <= -1) std::cout << "HAY! " << index << '\n';
		return vec[index];
	}

	void operator+=(Vector &other){
		for(int i=0; i<size; ++i) vec[i] += other.vec[i];
	}

	void operator-=(Vector &other){
		for(int i=0; i<size; ++i) vec[i] -= other.vec[i];
	}

	void operator*=(Vector &other){
		for(int i=0; i<size; ++i) vec[i] *= other.vec[i];
	}

	int length(){
		return size;
	}

	~Vector(){
		// std::cout << "Deleting: " << &vec << '\n';
		delete[] vec;
	}
};

class Matrix{
	Vector *mat = nullptr;
	int width, height;

public:

	Matrix(){}

	Matrix(int w, int h): width(w), height(h){
		mat = new Vector[height];
		for(int i=0; i<height; ++i) mat[i].make(width);
	}

	void make(int w, int h){
		delete[] mat;
		width = w;
		height = h;
		// std::cout << "Hey " << width << ", " << height << '\n';
		mat = new Vector[height];
		// std::cout << "Just made: " << &mat << '\n';
		for(int i=0; i<height; ++i) mat[i].make(width);
	}

	void fill(double val){
		for(int i=0; i<height; ++i) mat[i].fill(val);
	}

	void dot(Vector &vec){
		// if(vec.size() != width) return Vector();
		Vector in(vec);
		vec.make(height);
		for(int i=0; i<height; ++i) vec[i] = mat[i].dot(in);
	}

	Vector& operator[](int i){
		// if(i >= width || i <= -1) return Vector();
		return mat[i];
	}

	double& operator()(int y, int x){
		// if(y >= height || y <= -1) std::cout << "HAY! " << y << ", " << x << '\n';
		return mat[y][x];
	}

	void operator-=(Matrix &other){
		for(int i=0; i<height; ++i) mat[i] -= other.mat[i];
	}

	int yDim(){
		return height;
	}

	int xDim(){
		return width;
	}

	~Matrix(){
		// std::cout << "Deleting Matrix\n";
		delete[] mat;
	}
};

#endif