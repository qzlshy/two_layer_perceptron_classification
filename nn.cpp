#include "nn.h"

template<class T>
int Mtrx<T>::init(int a,int b) {
	int i;
	x=a;
	y=b;
	value=new T*[x];
	for(i=0;i<x;i++)
		value[i]=new T[y];
	vctr=value[0];
	return 1;
}

template<class T>
int Mtrx<T>::init(int a) {
	int i;
	x=1;
	y=a;
	value=new T*[x];
	for(i=0;i<x;i++)
		value[i]=new T[y];
	vctr=value[0];
	return 1;
}

template<class T>
int Mtrx<T>::free() {
	int i;
	for(i=0;i<x;i++)
		delete []value[i];
	delete []value;
	x=0;
	y=0;
}

template<class T>
int Mtrx<T>::assignment(T *m) {
	int i;
	for(i=0;i<y;i++)
		m[i]=value[0][i];
}

template<class T>
int Mtrx<T>::assignment(T **m) {
	int i,j;
	for(i=0;i<x;i++)
	for(j=0;j<y;j++)
		m[i][j]=value[i][j];
}

template<class T>
int Mtrx<T>::zeros() {
	int i,j;
	for(i=0;i<x;i++)
	for(j=0;j<y;j++)
		value[i][j]=0.0;
}

template<class T>
int Mtrx<T>::argmax() {
	int i,j,k1,k2;
	T t;
	t=value[0][0];
	k1=0;
	k2=0;
	for(i=0;i<x;i++)
	for(j=0;j<y;j++) {
		if(value[i][j]>t) {
			t=value[i][j];
			k1=i;
			k2=j;
		}
	}
	return k2;
}

template<class T>
Mtrx<T> Mtrx<T>::map(T (*fun)(T)) {
	int i,j;
	Mtrx<T> mr(x,y);
	for(i=0;i<x;i++)
	for(j=0;j<y;j++)
		mr.value[i][j]=fun(value[i][j]);
	return mr;
}

template<class T>
int Mtrx<T>::constant(T v) {
	int i,j;
	for(i=0;i<x;i++)
	for(j=0;j<y;j++)
		value[i][j]=v;
}

template<class T>
int Mtrx<T>::truncated_normal(T mean,T stddev) {
	int i,j;
	T t;
	std::default_random_engine gen(time(NULL)); 
	std::normal_distribution<T> normal(mean,stddev);
	for(i=0;i<x;i++)
	for(j=0;j<y;j++) {
		while(1) {
			t=normal(gen);
			if(fabs(t-mean)<=stddev*2.0) {
				value[i][j]=t;
				break;
			}
		}
	}
}

template<class T>
Mtrx<T> Mtrx<T>::transpose() {
	int i,j;
	Mtrx<T> mr(y,x);
	for(i=0;i<y;i++)
	for(j=0;j<x;j++)
		mr.value[i][j]=value[j][i];
	return mr;
}

template<class T>
void Mtrx<T>::operator<<(const T *v) {
	int i;
	for(i=0;i<y;i++)
		value[0][i]=v[i];
}

template<class T>
void Mtrx<T>::operator<<(const T **v) {
	int i,j;
	for(i=0;i<x;i++)
	for(j=0;j<y;j++)
		value[i][j]=v[i][j];
}

template<class T>
void Mtrx<T>::operator<<(const vector<T> &v) {
	int i;
	for(i=0;i<y;i++)
		value[0][i]=v[i];
}

template<class T>
void Mtrx<T>::operator<<(const vector< vector<T> > &v) {
	int i,j;
	for(i=0;i<x;i++)
	for(j=0;j<y;j++)
		value[i][j]=v[i][j];
}

template<class T>
void Mtrx<T>::operator=(const Mtrx<T> &m) {
	int i,j;
	if(x!=m.x) {
		cout<<"operator =: Mtrx dim not fit!\n";
		exit(0);
	}
	if(y!=m.y) {
		cout<<"operator =: Mtrx dim not fit!\n";
		exit(0);
	}
	for(i=0;i<x;i++)
	for(j=0;j<y;j++)
		value[i][j]=m.value[i][j];
}

template<class T>
Mtrx<T> Mtrx<T>::operator*(const Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(x,y);
	for(i=0;i<x;i++)
	for(j=0;j<y;j++) 
		mr.value[i][j]=value[i][j]*m.value[i][j];
	return mr;
}

template<class T>
Mtrx<T> Mtrx<T>::operator+(const Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(x,y);
	for(i=0;i<x;i++)
	for(j=0;j<y;j++)
		mr.value[i][j]=value[i][j]+m.value[i][j];
	return mr;
}

template<class T>
Mtrx<T> Mtrx<T>::operator-(const Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(x,y);
	for(i=0;i<x;i++)
	for(j=0;j<y;j++)
		mr.value[i][j]=value[i][j]-m.value[i][j];
	return mr;
}

template<class T>
Mtrx<T> Mtrx<T>::operator/(const Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(x,y);
	for(i=0;i<x;i++)
	for(j=0;j<y;j++) {
		if(value[i][j]==0.0)
			mr.value[i][j]=0.0;
		else
			mr.value[i][j]=value[i][j]/m.value[i][j];
	}
	return mr;
}

template<class T>
Mtrx<T> Mtrx<T>::operator*(T t) {
	int i,j;
	Mtrx<T> mr(x,y);
	for(i=0;i<x;i++)
	for(j=0;j<y;j++)
		mr.value[i][j]=value[i][j]*t;
	return mr;
}

template<class T>
Mtrx<T> Mtrx<T>::operator+(T t) {
	int i,j;
	Mtrx<T> mr(x,y);
	for(i=0;i<x;i++)
	for(j=0;j<y;j++)
		mr.value[i][j]=value[i][j]+t;
	return mr;
}

template<class T>
Mtrx<T> Mtrx<T>::operator-(T t) {
	int i,j;
	Mtrx<T> mr(x,y);
	for(i=0;i<x;i++)
	for(j=0;j<y;j++)
		mr.value[i][j]=value[i][j]-t;
	return mr;
}

template<class T>
Mtrx<T> Mtrx<T>::operator/(T t) {
	int i,j;
	Mtrx<T> mr(x,y);
	for(i=0;i<x;i++)
	for(j=0;j<y;j++)
		mr.value[i][j]=value[i][j]/t;
	return mr;
}

template<class T>
Mtrx<T> operator*(T t,const Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(m.x,m.y);
	for(i=0;i<m.x;i++)
	for(j=0;j<m.y;j++)
		mr.value[i][j]=t*m.value[i][j];
	return mr;
}

template<class T>
Mtrx<T> operator+(T t,const Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(m.x,m.y);
	for(i=0;i<m.x;i++)
	for(j=0;j<m.y;j++)
		mr.value[i][j]=t+m.value[i][j];
	return mr;
}

template<class T>
Mtrx<T> operator-(T t,const Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(m.x,m.y);
	for(i=0;i<m.x;i++)
	for(j=0;j<m.y;j++)
		mr.value[i][j]=t-m.value[i][j];
	return mr;
}

template<class T>
Mtrx<T> operator/(T t,const Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(m.x,m.y);
	for(i=0;i<m.x;i++)
	for(j=0;j<m.y;j++)
		mr.value[i][j]=t/m.value[i][j];
	return mr;
}

template<class T>
Mtrx<T> operator-(const Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(m.x,m.y);
	for(i=0;i<m.x;i++)
	for(j=0;j<m.y;j++)
		mr.value[i][j]=-m.value[i][j];
	return mr;
}

template<class T>
Mtrx<T> dot(T *t,const Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(m.y);
	T tmp;
	for(i=0;i<m.y;i++) {
		tmp=0;
		for(j=0;j<m.x;j++)
			tmp+=m.value[j][i]*t[j];
		mr.value[0][i]=tmp;
	}
	return mr;
}

template<class T>
Mtrx<T> dot(const vector<T> &t,const Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(m.y);
	T tmp;
	for(i=0;i<m.y;i++) {
		tmp=0;
		for(j=0;j<m.x;j++)
			tmp+=m.value[j][i]*t[j];
		mr.value[0][i]=tmp;
	}
	return mr;
}

template<class T>
Mtrx<T> dot(const Mtrx<T> &m1,const Mtrx<T> &m2) {
	int i1,i2,i3;
	T t;
	if(m1.y!=m2.x) {
		cout<<"operator *: Mtrx dim not fit!\n";
		exit(0);
	}
	Mtrx<T> mr(m1.x,m2.y);
	for(i1=0;i1<m1.x;i1++)
	for(i2=0;i2<m2.y;i2++) {
		t=0.0;
		for(i3=0;i3<m1.y;i3++)
			t+=m1.value[i1][i3]*m2.value[i3][i2];
		mr.value[i1][i2]=t;
	}
	return mr;
}

template<class T>
void operator<<(T *v,const Mtrx<T> &m) {
	int i;
	for(i=0;i<m.y;i++)
		v[i]=m.value[0][i];
}

template<class T>
void operator<<(T **v,const Mtrx<T> &m) {
	int i,j;
	for(i=0;i<m.x;i++)
	for(j=0;j<m.y;j++)
		v[i][j]=m.value[i][j];
}

template<class T>
Mtrx<T> log(const Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(m.x,m.y);
	for(i=0;i<m.x;i++)
	for(j=0;j<m.y;j++)
		mr.value[i][j]=log(m.value[i][j]);
	return mr;
}


template<class T>
int Layer<T>::init(int n1,int n2) {
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

template<class T>
int Layer<T>::forward(Mtrx<T>& a_b,Mtrx<T>& w,Mtrx<T>& b,Mtrx<T> (*actv)(Mtrx<T>&)) {
	z=dot(a_b,w)+b;
	a=actv(z);
}

template<class T>
int Layer<T>::forward(Mtrx<T>& a_b,Mtrx<T>& w,Mtrx<T>& b) {
	z=dot(a_b,w)+b;
	a=z;
}

template<class T>
int Layer<T>::backward(Mtrx<T>& delta_f,Mtrx<T>& w_f,Mtrx<T> (*actv)(Mtrx<T>&),Mtrx<T>& a_b) {
	d_ac_z=actv(z);
	delta=dot(delta_f,w_f.transpose())*d_ac_z;
	d_b=d_b+delta;
	d_w=d_w+dot(a_b.transpose(),delta);
	batch_size++;
}

template<class T>
int Layer<T>::backward(Mtrx<T>& delta_f,Mtrx<T>& a_b) {
	delta=delta_f;
	d_b=d_b+delta;
	d_w=d_w+dot(a_b.transpose(),delta);
	batch_size++;
}

template<class T>
int Layer<T>::backward(Mtrx<T>& delta_c,Mtrx<T> (*actv)(Mtrx<T>&),Mtrx<T>& a_b) {
	d_ac_z=actv(z);
	delta=delta_c*d_ac_z;
	d_b=d_b+delta;
	d_w=d_w+dot(a_b.transpose(),delta);
	batch_size++;
}

template<class T>
int Layer<T>::backward(Mtrx<T>& delta_c,Mtrx<T> (*actv)(Mtrx<T>&,Mtrx<T>&),Mtrx<T>& a_b) {
	delta=actv(a,delta_c);
	d_b=d_b+delta;
	d_w=d_w+dot(a_b.transpose(),delta);
	batch_size++;
}

template<class T>
int Layer<T>::GD(Mtrx<T>& w,Mtrx<T>& b,T learning_rate) {
	w=w-learning_rate*d_w/T(batch_size);
	b=b-learning_rate*d_b/T(batch_size);
	batch_size=0;
	d_w.zeros();
	d_b.zeros();
}

template<class T>
Mtrx<T> relu(Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(m.x,m.y);
	for(i=0;i<m.x;i++)
	for(j=0;j<m.y;j++) {
		if(m.value[i][j]<=0)
			mr.value[i][j]=0;
		else
			mr.value[i][j]=m.value[i][j];
	}
	return mr;
}

template<class T>
Mtrx<T> relu_dz(Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(m.x,m.y);
	for(i=0;i<m.x;i++)
	for(j=0;j<m.y;j++) {
		if(m.value[i][j]<=0)
			mr.value[i][j]=0;
		else
			mr.value[i][j]=1;
	}
	return mr;
}

template<class T>
Mtrx<T> softmax(Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(m.x,m.y);
	double t=0.0;
	for(i=0;i<m.x;i++)
	for(j=0;j<m.y;j++) {
		t+=exp(double(m.value[i][j]));
	}
	for(i=0;i<m.x;i++)
	for(j=0;j<m.y;j++) {
		mr.value[i][j]=exp(double(m.value[i][j]))/t;
	}
	return mr;
}

template<class T>
Mtrx<T> softmax_cross_entropy_dz(Mtrx<T> &m1,Mtrx<T> &m2) {
	int i,j;
	Mtrx<T> mr(m1.x,m1.y);
	for(i=0;i<m1.x;i++)
	for(j=0;j<m1.y;j++) {
		if(m2.value[i][j]==1.0)
			mr.value[i][j]=m1.value[i][j]-1.0;
		else
			mr.value[i][j]=m1.value[i][j];
	}
	return(mr);
}

template<class T>
Mtrx<T> softmax_dz(Mtrx<T> &m) {
	int i,j;
	Mtrx<T> mr(m.x,m.y);
	Mtrx<T> m_a(m.x,m.y);
	Mtrx<T> m_d(m.y,m.y);
	m_a=softmax(m);
	for(i=0;i<m.y;i++)
	for(j=0;j<m.y;j++) {
		if(i==j)
			m_d.value[i][j]=m_a.value[0][j]*(1-m_a.value[0][j]);
		else
			m_d.value[i][j]=-m_a.value[0][j]*m_a.value[0][i];
	}
	return m_d;
}
