#include "nn.cpp"
#include "readmnist.h"



int main(int argc,char **argv) {
	int i,j,k;
	vector<float> labels;
	vector< vector<float> > images;
	vector<float> test_labels;
	vector< vector<float> > test_images;
	read_Mnist_Images(argv[1], images);
	read_Mnist_Label(argv[2], labels);
	read_Mnist_Images(argv[3], test_images);
	read_Mnist_Label(argv[4], test_labels);

	int labels_num=labels.size();
	int images_num=images.size();
	int image_size=images.at(0).size();


	int batch_size=100;
	float learn_rate=0.01;
	float mean=0.0;
	float stddev=0.1;
	float loss;
	
	Mtrx<float> w1(image_size,100);
	Mtrx<float> b1(100);

	w1.truncated_normal(mean,stddev);
	b1.truncated_normal(mean,stddev);

	Mtrx<float> w2(100,10);
	Mtrx<float> b2(10);

	w2.truncated_normal(mean,stddev);
	b2.truncated_normal(mean,stddev);


	Mtrx<float> train_images(image_size);
	Mtrx<float> train_labels(10);
	Mtrx<float> loss_da(10);

	Layer<float> L1(image_size,100);
	Layer<float> L2(100,10);

	int n1,n2,r1,t1;

	int batch_num=int(images_num/batch_size);
	vector< vector<float> > data_images;
	vector<float> data_labels;

	int round;
	for(round=0;round<10;round++) {
	for(i=0;i<batch_num;i++) {
		data_images.insert(data_images.begin(),images.begin()+i*batch_size, images.begin()+(i+1)*batch_size);
		data_labels.insert(data_labels.begin(),labels.begin()+i*batch_size, labels.begin()+(i+1)*batch_size);
		loss=0.0;
		r1=0; t1=0;
		for(j=0;j<batch_size;j++) {
			train_images<<data_images[j];
			train_images=train_images/255.0;
			train_labels.zeros();
			train_labels[0][int(data_labels[j])]=1.0;
			L1.forward(train_images,w1,b1,relu);
			L2.forward(L1.a,w2,b2,softmax);
			loss+=-(dot(train_labels,(log(L2.a)).transpose())).value[0][0];
			n1=train_labels.argmax();
			n2=L2.a.argmax();
			if(n1==n2)
				r1++;
			t1++;
			L2.backward(train_labels,softmax_cross_entropy_dz,L1.a);
			L1.backward(L2.delta,w2,relu_dz,train_images);
		}
		loss/=float(batch_size);
//		cout<<"loss"<<' '<<loss<<' '<<float(r1)/float(t1)<<endl;
		L1.GD(w1,b1,learn_rate);
		L2.GD(w2,b2,learn_rate);
		data_images.clear();
		data_labels.clear();
	}

	
	int test_num=test_images.size();
	int test_size=test_images.at(0).size();
	Mtrx<float> t_images(test_size);
	Mtrx<float> t_labels(10);
	r1=0; t1=0;
	for(i=0;i<test_num;i++) {
		t_images<<test_images[i];
		t_images=t_images/255.0;
		t_labels.zeros();
		t_labels[0][int(test_labels[i])]=1.0;
		L1.forward(t_images,w1,b1,relu);
		L2.forward(L1.a,w2,b2,softmax);
		n1=t_labels.argmax();
		n2=L2.a.argmax();
		if(n1==n2)
			r1++;
		t1++;
	}
	cout<<"reight: "<<float(r1)/float(t1)<<endl;
	}
	
}
