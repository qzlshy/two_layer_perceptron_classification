#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cstdlib>
#include <random>
#define _USE_MATH_DEFINES
#include <cmath>
#include <string.h>
#include <stdio.h>
#include <vector>

using namespace std;

template<class T>
class Mtrx {
	public:
	int x,y;
	Mtrx(int a,int b) {
		int i;
		x=a;
		y=b;
		value=new T*[x];
		for(i=0;i<x;i++)
			value[i]=new T[y];
		vctr=value[0];
	}
	Mtrx(int a) {
		int i;
		x=1;
		y=a;
		value=new T*[x];
		for(i=0;i<x;i++)
			value[i]=new T[y];
		vctr=value[0];
	}
	Mtrx() {
	}
	~Mtrx() {
		int i;
		for(i=0;i<x;i++)
			delete []value[i];
		delete []value;
		x=0;
		y=0;
	}
	T **value;
	T *vctr;
	int init(int,int);
	int init(int);
	int free();
	int assignment(T *);
	int assignment(T **);
	int zeros();
	int argmax();
	int constant(T);
	int truncated_normal(T,T);
	Mtrx map(T (*fun)(T));
	T* operator[](int k) {
		return value[k];
	}
	Mtrx transpose();
	void operator<<(const T *);
	void operator<<(const T **);
	void operator<<(const vector<T> &);
	void operator<<(const vector< vector<T> > &);
	void operator=(const Mtrx &);
	Mtrx operator*(const Mtrx &);
	Mtrx operator+(const Mtrx &);
	Mtrx operator-(const Mtrx &);
	Mtrx operator/(const Mtrx &);
	Mtrx operator*(T);
	Mtrx operator+(T);
	Mtrx operator-(T);
	Mtrx operator/(T);
	template<class S>
	friend Mtrx<S> operator*(S,const Mtrx<S> &);
	template<class S>
	friend Mtrx<S> operator+(S,const Mtrx<S> &);
	template<class S>
	friend Mtrx<S> operator-(S,const Mtrx<S> &);
	template<class S>
	friend Mtrx<S> operator/(S,const Mtrx<S> &);
	template<class S>
	friend Mtrx<S> operator-(const Mtrx<S> &);
	template<class S>
	friend Mtrx<S> dot(S *,const Mtrx<S> &);
	template<class S>
	friend Mtrx<S> dot(const vector<S> &,const Mtrx<S> &);
	template<class S>
	friend Mtrx<S> dot(const Mtrx<S> &,const Mtrx<S> &);
	template<class S>
	friend void operator<<(S *, const Mtrx<S> &);
	template<class S>
	friend void operator<<(S **, const Mtrx<S> &);
};

template<class T>
class Layer {
	public:
	int batch_size;
	int n;
	int m;
	Mtrx<T> z;
	Mtrx<T> d_ac_z;
	Mtrx<T> a;
	Mtrx<T> delta;
	Mtrx<T> d_w;
	Mtrx<T> d_b;
	Layer(int n1,int n2) {
		m=n1;
		n=n2;
		batch_size=0;
		z.init(n);
		d_ac_z.init(n);
		a.init(n);
		delta.init(n);
		d_w.init(m,n);
		d_b.init(n);
		d_w.zeros();
		d_b.zeros();
	}
	int init(int,int);
	int forward(Mtrx<T>&,Mtrx<T>&,Mtrx<T>&,Mtrx<T> (*actv)(Mtrx<T>&));
	int forward(Mtrx<T>&,Mtrx<T>&,Mtrx<T>&);
	int backward(Mtrx<T>&,Mtrx<T>&);
	int backward(Mtrx<T>&,Mtrx<T>&,Mtrx<T> (*actv)(Mtrx<T>&),Mtrx<T>&);
	int backward(Mtrx<T>&,Mtrx<T> (*actv)(Mtrx<T>&),Mtrx<T>&);
	int backward(Mtrx<T>&,Mtrx<T> (*actv)(Mtrx<T>&,Mtrx<T>&),Mtrx<T>&);
	int GD(Mtrx<T>&,Mtrx<T>&,T);
};

template<class T>
Mtrx<T> relu(Mtrx<T> &);
template<class T>
Mtrx<T> relu_dz(Mtrx<T> &);
template<class T>
Mtrx<T> softmax(Mtrx<T> &);
template<class T>
Mtrx<T> softmax_dz(Mtrx<T> &);
template<class T>
Mtrx<T> softmax_cross_entropy_dz(Mtrx<T> &,Mtrx<T> &);
