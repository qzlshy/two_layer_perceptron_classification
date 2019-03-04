#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;


int ReverseInt(int);
void read_Mnist_Label(const char *, vector<float>&);
void read_Mnist_Images(const char *, vector< vector<float> >&);
